//! Matrix linear algebra and geospatial function dispatch
//!
//! Matrix cells are Binary in the zyron_types::matrix encoding of
//! [u32 rows LE][u32 cols LE][f64 row-major]
//! Geometry cells are Binary in the zyron_types::geospatial encode_wkb form
//! Vector arguments and Array results are JSON number arrays as text bytes
//! inside Binary cells
//! Composite results (svd, pca) are JSON objects as text bytes because the
//! backing zyron-types functions return tuples with no byte serialization

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_types::geospatial::{self, Geometry};
use zyron_types::matrix;

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "cross_product" => cross_product_impl(args),
        "dot_product" => dot_product_impl(args),
        "eigenvalues" => eigenvalues_impl(args),
        "matrix_add" => matrix_bin2("matrix_add(matrix, matrix)", args, matrix::matrix_add),
        "matrix_create" => matrix_create_impl(args),
        "matrix_determinant" => matrix_f64(
            "matrix_determinant(matrix)",
            args,
            matrix::matrix_determinant,
        ),
        "matrix_identity" => matrix_identity_impl(args),
        "matrix_inverse" => matrix_bin1("matrix_inverse(matrix)", args, matrix::matrix_inverse),
        "matrix_multiply" => matrix_bin2(
            "matrix_multiply(matrix, matrix)",
            args,
            matrix::matrix_multiply,
        ),
        "matrix_norm" => matrix_norm_impl(args),
        "matrix_scalar_multiply" => matrix_scalar_multiply_impl(args),
        "matrix_subtract" => matrix_bin2(
            "matrix_subtract(matrix, matrix)",
            args,
            matrix::matrix_subtract,
        ),
        "matrix_trace" => matrix_f64("matrix_trace(matrix)", args, matrix::matrix_trace),
        "matrix_transpose" => {
            matrix_bin1("matrix_transpose(matrix)", args, matrix::matrix_transpose)
        }
        "pca" => pca_impl(args),
        "svd" => svd_impl(args),
        "h3_distance" => h3_distance_impl(args),
        "h3_from_point" => h3_from_point_impl(args),
        "h3_to_boundary" => h3_to_boundary_impl(args),
        "st_area" => geo_f64_1("st_area(geometry)", args, geospatial::st_area),
        "st_as_geojson" => geo_text1("st_as_geojson(geometry)", args, |g| {
            geospatial::st_as_geojson(g)
        }),
        "st_as_text" => geo_text1("st_as_text(geometry)", args, |g| geospatial::st_as_text(g)),
        "st_buffer" => st_buffer_impl(args),
        "st_centroid" => geo_geom1("st_centroid(geometry)", args, geospatial::st_centroid),
        "st_contains" => geo_bool2(
            "st_contains(geometry, geometry)",
            args,
            geospatial::st_contains,
        ),
        "st_distance" => geo_f64_2(
            "st_distance(geometry, geometry)",
            args,
            geospatial::st_distance,
        ),
        "st_dwithin" => st_dwithin_impl(args),
        "st_geom_from_geojson" => parse_geom(
            "st_geom_from_geojson(text)",
            args,
            geospatial::st_geom_from_geojson,
        ),
        "st_geom_from_text" => parse_geom(
            "st_geom_from_text(text)",
            args,
            geospatial::st_geom_from_text,
        ),
        "st_intersects" => geo_bool2(
            "st_intersects(geometry, geometry)",
            args,
            geospatial::st_intersects,
        ),
        "st_make_point" => st_make_point_impl(args),
        "st_union" => geo_geom2("st_union(geometry, geometry)", args, geospatial::st_union),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Argument extraction, every error carries the function signature
// ---------------------------------------------------------------------------

fn check_args(sig: &str, args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} arguments",
            sig, expected
        )));
    }
    Ok(())
}

fn bytes_arg<'a>(sig: &str, col: &'a Column, pos: usize) -> Result<Vec<&'a [u8]>> {
    super::column_bytes(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} argument {} must be binary", sig, pos)))
}

fn strs_arg<'a>(sig: &str, col: &'a Column, pos: usize) -> Result<Vec<&'a str>> {
    super::column_strings(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} argument {} must be text", sig, pos)))
}

fn floats_arg(sig: &str, col: &Column, pos: usize) -> Result<Vec<f64>> {
    super::column_floats(col).map_err(|_| {
        ZyronError::ExecutionError(format!("{} argument {} must be numeric", sig, pos))
    })
}

fn ints_arg(sig: &str, col: &Column, pos: usize) -> Result<Vec<i64>> {
    super::column_ints(col).map_err(|_| {
        ZyronError::ExecutionError(format!("{} argument {} must be an integer", sig, pos))
    })
}

fn row_err(sig: &str, e: ZyronError) -> ZyronError {
    ZyronError::ExecutionError(format!("{}: {}", sig, e))
}

fn any_null(args: &[Column], i: usize) -> bool {
    args.iter().any(|c| c.nulls.is_null(i))
}

// ---------------------------------------------------------------------------
// JSON helpers for vector args, Array results, and Composite results
// ---------------------------------------------------------------------------

// vector cells hold JSON number arrays as text bytes
fn json_f64_array(sig: &str, bytes: &[u8]) -> Result<Vec<f64>> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a JSON number array", sig)))?;
    let arr = parsed.as_array().ok_or_else(|| {
        ZyronError::ExecutionError(format!("{} expects a JSON number array", sig))
    })?;
    arr.iter()
        .map(|v| {
            v.as_f64().ok_or_else(|| {
                ZyronError::ExecutionError(format!("{} expects numeric array elements", sig))
            })
        })
        .collect()
}

// Value::from maps non-finite floats to JSON null without panicking
fn f64s_to_json(values: &[f64]) -> serde_json::Value {
    serde_json::Value::Array(values.iter().map(|&v| serde_json::Value::from(v)).collect())
}

fn f64s_to_json_bytes(values: &[f64]) -> Vec<u8> {
    f64s_to_json(values).to_string().into_bytes()
}

// decodes an encoded matrix into a JSON object with rows, cols, data fields
fn matrix_json(sig: &str, bytes: &[u8]) -> Result<serde_json::Value> {
    let (rows, cols, data) = matrix::matrix_decode(bytes).map_err(|e| row_err(sig, e))?;
    let mut obj = serde_json::Map::new();
    obj.insert("rows".to_string(), serde_json::Value::from(rows));
    obj.insert("cols".to_string(), serde_json::Value::from(cols));
    obj.insert("data".to_string(), f64s_to_json(&data));
    Ok(serde_json::Value::Object(obj))
}

// ---------------------------------------------------------------------------
// Matrix combinators, domain failures surface as ExecutionError because
// dimension mismatches and singular inputs are structurally invalid calls
// ---------------------------------------------------------------------------

fn matrix_bin1(sig: &str, args: &[Column], f: impl Fn(&[u8]) -> Result<Vec<u8>>) -> Result<Column> {
    check_args(sig, args, 1)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let n = m.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(f(m[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Matrix,
    ))
}

fn matrix_bin2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&[u8], &[u8]) -> Result<Vec<u8>>,
) -> Result<Column> {
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(f(a[i], b[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Matrix,
    ))
}

fn matrix_f64(sig: &str, args: &[Column], f: impl Fn(&[u8]) -> Result<f64>) -> Result<Column> {
    check_args(sig, args, 1)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let n = m.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        data.push(f(m[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// ---------------------------------------------------------------------------
// Matrix implementations
// ---------------------------------------------------------------------------

fn cross_product_impl(args: &[Column]) -> Result<Column> {
    let sig = "cross_product(array, array)";
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let va = json_f64_array(sig, a[i])?;
        let vb = json_f64_array(sig, b[i])?;
        let out = matrix::cross_product(&va, &vb).map_err(|e| row_err(sig, e))?;
        data.push(f64s_to_json_bytes(&out));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn dot_product_impl(args: &[Column]) -> Result<Column> {
    let sig = "dot_product(array, array)";
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let va = json_f64_array(sig, a[i])?;
        let vb = json_f64_array(sig, b[i])?;
        data.push(matrix::dot_product(&va, &vb).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn eigenvalues_impl(args: &[Column]) -> Result<Column> {
    let sig = "eigenvalues(matrix)";
    check_args(sig, args, 1)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let n = m.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let eigs = matrix::eigenvalues(m[i]).map_err(|e| row_err(sig, e))?;
        data.push(f64s_to_json_bytes(&eigs));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn matrix_create_impl(args: &[Column]) -> Result<Column> {
    let sig = "matrix_create(int, int, array)";
    check_args(sig, args, 3)?;
    let rows = ints_arg(sig, &args[0], 1)?;
    let cols = ints_arg(sig, &args[1], 2)?;
    let vals = bytes_arg(sig, &args[2], 3)?;
    let n = rows.len().min(cols.len()).min(vals.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let r = u32::try_from(rows[i])
            .map_err(|_| ZyronError::ExecutionError(format!("{} rows out of range", sig)))?;
        let c = u32::try_from(cols[i])
            .map_err(|_| ZyronError::ExecutionError(format!("{} cols out of range", sig)))?;
        let v = json_f64_array(sig, vals[i])?;
        data.push(matrix::matrix_create(r, c, &v).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Matrix,
    ))
}

fn matrix_identity_impl(args: &[Column]) -> Result<Column> {
    let sig = "matrix_identity(int)";
    check_args(sig, args, 1)?;
    let sizes = ints_arg(sig, &args[0], 1)?;
    let n = sizes.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        // size capped at 4096 to bound the n by n f64 allocation
        if !(0..=4096).contains(&sizes[i]) {
            return Err(ZyronError::ExecutionError(format!(
                "{} size must be between 0 and 4096, got {}",
                sig, sizes[i]
            )));
        }
        data.push(matrix::matrix_identity(sizes[i] as u32));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Matrix,
    ))
}

fn matrix_norm_impl(args: &[Column]) -> Result<Column> {
    let sig = "matrix_norm(matrix, text)";
    check_args(sig, args, 2)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let kinds = strs_arg(sig, &args[1], 2)?;
    let n = m.len().min(kinds.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        data.push(matrix::matrix_norm(m[i], kinds[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn matrix_scalar_multiply_impl(args: &[Column]) -> Result<Column> {
    let sig = "matrix_scalar_multiply(matrix, float)";
    check_args(sig, args, 2)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let scalars = floats_arg(sig, &args[1], 2)?;
    let n = m.len().min(scalars.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(matrix::matrix_scalar_multiply(m[i], scalars[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Matrix,
    ))
}

// composite cell is a JSON object with components, scores, variance_explained
fn pca_impl(args: &[Column]) -> Result<Column> {
    let sig = "pca(matrix, int)";
    check_args(sig, args, 2)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let comps = ints_arg(sig, &args[1], 2)?;
    let n = m.len().min(comps.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let k = u32::try_from(comps[i])
            .map_err(|_| ZyronError::ExecutionError(format!("{} components out of range", sig)))?;
        let (components, scores, variance) = matrix::pca(m[i], k).map_err(|e| row_err(sig, e))?;
        let mut obj = serde_json::Map::new();
        obj.insert("components".to_string(), matrix_json(sig, &components)?);
        obj.insert("scores".to_string(), matrix_json(sig, &scores)?);
        obj.insert("variance_explained".to_string(), f64s_to_json(&variance));
        data.push(serde_json::Value::Object(obj).to_string().into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

// composite cell is a JSON object with u, s, vt matrices
fn svd_impl(args: &[Column]) -> Result<Column> {
    let sig = "svd(matrix)";
    check_args(sig, args, 1)?;
    let m = bytes_arg(sig, &args[0], 1)?;
    let n = m.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (u, s, vt) = matrix::svd(m[i]).map_err(|e| row_err(sig, e))?;
        let mut obj = serde_json::Map::new();
        obj.insert("u".to_string(), matrix_json(sig, &u)?);
        obj.insert("s".to_string(), matrix_json(sig, &s)?);
        obj.insert("vt".to_string(), matrix_json(sig, &vt)?);
        data.push(serde_json::Value::Object(obj).to_string().into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

// ---------------------------------------------------------------------------
// Geometry combinators, cells decode through decode_wkb, malformed cells are
// structurally invalid and surface as ExecutionError
// ---------------------------------------------------------------------------

fn decode_geom(sig: &str, bytes: &[u8]) -> Result<Geometry> {
    geospatial::decode_wkb(bytes).map_err(|e| row_err(sig, e))
}

fn geo_f64_1(sig: &str, args: &[Column], f: impl Fn(&Geometry) -> Result<f64>) -> Result<Column> {
    check_args(sig, args, 1)?;
    let g = bytes_arg(sig, &args[0], 1)?;
    let n = g.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let geom = decode_geom(sig, g[i])?;
        data.push(f(&geom).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn geo_f64_2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&Geometry, &Geometry) -> Result<f64>,
) -> Result<Column> {
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let ga = decode_geom(sig, a[i])?;
        let gb = decode_geom(sig, b[i])?;
        data.push(f(&ga, &gb).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn geo_bool2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&Geometry, &Geometry) -> Result<bool>,
) -> Result<Column> {
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(false);
            nulls.set_null(i);
            continue;
        }
        let ga = decode_geom(sig, a[i])?;
        let gb = decode_geom(sig, b[i])?;
        data.push(f(&ga, &gb).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn geo_text1(sig: &str, args: &[Column], f: impl Fn(&Geometry) -> String) -> Result<Column> {
    check_args(sig, args, 1)?;
    let g = bytes_arg(sig, &args[0], 1)?;
    let n = g.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let geom = decode_geom(sig, g[i])?;
        data.push(f(&geom));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Text,
    ))
}

fn geo_geom1(
    sig: &str,
    args: &[Column],
    f: impl Fn(&Geometry) -> Result<Geometry>,
) -> Result<Column> {
    check_args(sig, args, 1)?;
    let g = bytes_arg(sig, &args[0], 1)?;
    let n = g.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let geom = decode_geom(sig, g[i])?;
        let out = f(&geom).map_err(|e| row_err(sig, e))?;
        data.push(geospatial::encode_wkb(&out));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

fn geo_geom2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&Geometry, &Geometry) -> Result<Geometry>,
) -> Result<Column> {
    check_args(sig, args, 2)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let ga = decode_geom(sig, a[i])?;
        let gb = decode_geom(sig, b[i])?;
        let out = f(&ga, &gb).map_err(|e| row_err(sig, e))?;
        data.push(geospatial::encode_wkb(&out));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

// parser style constructor, an unparseable row yields NULL matching the
// validator convention used by nearby bridge arms
fn parse_geom(sig: &str, args: &[Column], f: impl Fn(&str) -> Result<Geometry>) -> Result<Column> {
    check_args(sig, args, 1)?;
    let texts = strs_arg(sig, &args[0], 1)?;
    let n = texts.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        match f(texts[i]) {
            Ok(g) => data.push(geospatial::encode_wkb(&g)),
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

// ---------------------------------------------------------------------------
// Geospatial implementations with non uniform argument shapes
// ---------------------------------------------------------------------------

fn st_make_point_impl(args: &[Column]) -> Result<Column> {
    let sig = "st_make_point(float, float)";
    check_args(sig, args, 2)?;
    let lon = floats_arg(sig, &args[0], 1)?;
    let lat = floats_arg(sig, &args[1], 2)?;
    let n = lon.len().min(lat.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(geospatial::encode_wkb(&geospatial::st_make_point(
            lon[i], lat[i],
        )));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

fn st_buffer_impl(args: &[Column]) -> Result<Column> {
    let sig = "st_buffer(geometry, float)";
    check_args(sig, args, 2)?;
    let g = bytes_arg(sig, &args[0], 1)?;
    let dist = floats_arg(sig, &args[1], 2)?;
    let n = g.len().min(dist.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let geom = decode_geom(sig, g[i])?;
        let out = geospatial::st_buffer(&geom, dist[i]).map_err(|e| row_err(sig, e))?;
        data.push(geospatial::encode_wkb(&out));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

fn st_dwithin_impl(args: &[Column]) -> Result<Column> {
    let sig = "st_dwithin(geometry, geometry, float)";
    check_args(sig, args, 3)?;
    let a = bytes_arg(sig, &args[0], 1)?;
    let b = bytes_arg(sig, &args[1], 2)?;
    let radius = floats_arg(sig, &args[2], 3)?;
    let n = a.len().min(b.len()).min(radius.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(false);
            nulls.set_null(i);
            continue;
        }
        let ga = decode_geom(sig, a[i])?;
        let gb = decode_geom(sig, b[i])?;
        data.push(geospatial::st_dwithin(&ga, &gb, radius[i]).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn h3_from_point_impl(args: &[Column]) -> Result<Column> {
    let sig = "h3_from_point(float, float, int)";
    check_args(sig, args, 3)?;
    let lon = floats_arg(sig, &args[0], 1)?;
    let lat = floats_arg(sig, &args[1], 2)?;
    let res = ints_arg(sig, &args[2], 3)?;
    let n = lon.len().min(lat.len()).min(res.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        let r = u8::try_from(res[i])
            .map_err(|_| ZyronError::ExecutionError(format!("{} resolution must be 0-15", sig)))?;
        let idx = geospatial::h3_from_point(lon[i], lat[i], r).map_err(|e| row_err(sig, e))?;
        // index bit pattern flows through Int64, the cast preserves bits
        data.push(idx as i64);
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Int64,
    ))
}

fn h3_to_boundary_impl(args: &[Column]) -> Result<Column> {
    let sig = "h3_to_boundary(int)";
    check_args(sig, args, 1)?;
    let idx = ints_arg(sig, &args[0], 1)?;
    let n = idx.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        // bit preserving cast back to the packed u64 index
        let boundary = geospatial::h3_to_boundary(idx[i] as u64).map_err(|e| row_err(sig, e))?;
        data.push(geospatial::encode_wkb(&boundary));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Geometry,
    ))
}

fn h3_distance_impl(args: &[Column]) -> Result<Column> {
    let sig = "h3_distance(int, int)";
    check_args(sig, args, 2)?;
    let a = ints_arg(sig, &args[0], 1)?;
    let b = ints_arg(sig, &args[1], 2)?;
    let n = a.len().min(b.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        // bit preserving casts back to the packed u64 indexes
        data.push(geospatial::h3_distance(a[i] as u64, b[i] as u64).map_err(|e| row_err(sig, e))?);
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_types::matrix::{matrix_decode, matrix_encode};

    fn matrix_col(cells: Vec<Vec<u8>>) -> Column {
        Column::new(ColumnData::Binary(cells), TypeId::Matrix)
    }

    fn geom_col(cells: Vec<Vec<u8>>) -> Column {
        Column::new(ColumnData::Binary(cells), TypeId::Geometry)
    }

    #[test]
    fn matrix_add_adds_elementwise() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = matrix_encode(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = dispatch("matrix_add", &[matrix_col(vec![a]), matrix_col(vec![b])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(out.type_id, TypeId::Matrix);
        match &out.data {
            ColumnData::Binary(v) => {
                let (r, c, d) = matrix_decode(&v[0]).unwrap();
                assert_eq!((r, c), (2, 2));
                assert_eq!(d, vec![6.0, 8.0, 10.0, 12.0]);
            }
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    #[test]
    fn matrix_determinant_returns_float() {
        let m = matrix_encode(2, 2, &[3.0, 8.0, 4.0, 6.0]).unwrap();
        let out = dispatch("matrix_determinant", &[matrix_col(vec![m])], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Float64(v) => assert!((v[0] - (-14.0)).abs() < 1e-10),
            other => panic!("expected float column, got {:?}", other),
        }
    }

    #[test]
    fn matrix_add_null_row_propagates() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = matrix_encode(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let ca = Column::with_nulls(
            ColumnData::Binary(vec![a, Vec::new()]),
            nulls,
            TypeId::Matrix,
        );
        let cb = matrix_col(vec![b.clone(), b]);
        let out = dispatch("matrix_add", &[ca, cb], 2).unwrap().unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn cross_product_of_unit_vectors() {
        let a = Column::new(ColumnData::Binary(vec![b"[1,0,0]".to_vec()]), TypeId::Array);
        let b = Column::new(ColumnData::Binary(vec![b"[0,1,0]".to_vec()]), TypeId::Array);
        let out = dispatch("cross_product", &[a, b], 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Array);
        match &out.data {
            ColumnData::Binary(v) => {
                let parsed: Vec<f64> = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(parsed, vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    #[test]
    fn st_distance_nyc_to_london() {
        let nyc = geospatial::encode_wkb(&geospatial::st_make_point(-73.9857, 40.7484));
        let london = geospatial::encode_wkb(&geospatial::st_make_point(-0.1278, 51.5074));
        let out = dispatch(
            "st_distance",
            &[geom_col(vec![nyc]), geom_col(vec![london])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Float64(v) => assert!((v[0] - 5_570_000.0).abs() < 50_000.0),
            other => panic!("expected float column, got {:?}", other),
        }
    }

    #[test]
    fn st_geom_from_text_invalid_row_is_null() {
        let c = Column::new(
            ColumnData::Utf8(vec!["POINT(1 2)".to_string(), "nonsense".to_string()]),
            TypeId::Text,
        );
        let out = dispatch("st_geom_from_text", &[c], 2).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Geometry);
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::Binary(v) => {
                let g = geospatial::decode_wkb(&v[0]).unwrap();
                assert_eq!(g, geospatial::st_make_point(1.0, 2.0));
            }
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    #[test]
    fn h3_from_point_index_distance_to_itself_is_zero() {
        let lon = Column::new(ColumnData::Float64(vec![0.0]), TypeId::Float64);
        let lat = Column::new(ColumnData::Float64(vec![0.0]), TypeId::Float64);
        let res = Column::new(ColumnData::Int64(vec![5]), TypeId::Int64);
        let idx_col = dispatch("h3_from_point", &[lon, lat, res], 1)
            .unwrap()
            .unwrap();
        let idx = match &idx_col.data {
            ColumnData::Int64(v) => v[0],
            other => panic!("expected int64 column, got {:?}", other),
        };
        let a = Column::new(ColumnData::Int64(vec![idx]), TypeId::Int64);
        let b = Column::new(ColumnData::Int64(vec![idx]), TypeId::Int64);
        let out = dispatch("h3_distance", &[a, b], 1).unwrap().unwrap();
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[0], 0),
            other => panic!("expected int32 column, got {:?}", other),
        }
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("not_a_function", &[], 1).is_none());
    }
}
