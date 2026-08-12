//! Dispatch arms for money, quantity, and formatting functions
//!
//! Money cells are 10 bytes, i64 LE minor units then u16 LE ISO 4217 numeric
//! code, matching the storage layout documented in zyron_types::money
//! Quantity cells are 10 bytes, f64 LE value then u16 LE unit id, matching
//! zyron_types::quantity

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "currency_lookup" => currency_lookup_impl(args),
        "currency_by_numeric" => currency_by_numeric_impl(args),
        "money_create" => money_create_impl(args),
        "money_add" => money_binop_impl(
            "money_add",
            "money_add(money, money)",
            args,
            zyron_types::money::money_add,
        ),
        "money_subtract" => money_binop_impl(
            "money_subtract",
            "money_subtract(money, money)",
            args,
            zyron_types::money::money_subtract,
        ),
        "money_multiply" => money_multiply_impl(args),
        "money_convert" => money_convert_impl(args),
        "money_format" => money_format_impl(args),
        "money_currency_code" => currency_str_impl("money_currency_code", args, |c| {
            zyron_types::money::money_currency_code(c).to_string()
        }),
        "money_currency_symbol" => currency_str_impl("money_currency_symbol", args, |c| {
            zyron_types::money::money_currency_symbol(c).to_string()
        }),
        "money_minor_digits" => money_minor_digits_impl(args),
        "quantity_create" => quantity_create_impl(args),
        "quantity_add" => quantity_binop_impl(
            "quantity_add",
            "quantity_add(quantity, quantity)",
            args,
            zyron_types::quantity::quantity_add,
        ),
        "quantity_subtract" => quantity_binop_impl(
            "quantity_subtract",
            "quantity_subtract(quantity, quantity)",
            args,
            zyron_types::quantity::quantity_subtract,
        ),
        "quantity_multiply" => quantity_binop_impl(
            "quantity_multiply",
            "quantity_multiply(quantity, quantity)",
            args,
            zyron_types::quantity::quantity_multiply,
        ),
        "quantity_scale" => quantity_scale_impl(args),
        "quantity_convert" => quantity_convert_impl(args),
        "quantity_format" => quantity_format_impl(args),
        "quantity_dimension" => unit_str_impl("quantity_dimension", args, |u| {
            zyron_types::quantity::quantity_dimension(u).to_string()
        }),
        "quantity_unit_name" => unit_str_impl("quantity_unit_name", args, |u| {
            zyron_types::quantity::quantity_unit_name(u).to_string()
        }),
        "convert_units" => convert_units_impl(args),
        "format_currency" => format_currency_impl(args),
        "format_number" => format_number_impl(args),
        "format_percentage" => format_percentage_impl(args),
        "parse_number" => parse_number_impl(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// cell encoding
// ---------------------------------------------------------------------------

fn encode_money(minor_units: i64, currency: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    out.extend_from_slice(&minor_units.to_le_bytes());
    out.extend_from_slice(&currency.to_le_bytes());
    out
}

fn decode_money(name: &str, cell: &[u8]) -> Result<(i64, u16)> {
    if cell.len() != 10 {
        return Err(ZyronError::ExecutionError(format!(
            "{}: money value must be 10 bytes, got {}",
            name,
            cell.len()
        )));
    }
    let mut val = [0u8; 8];
    val.copy_from_slice(&cell[0..8]);
    let mut cur = [0u8; 2];
    cur.copy_from_slice(&cell[8..10]);
    Ok((i64::from_le_bytes(val), u16::from_le_bytes(cur)))
}

fn encode_quantity(value: f64, unit: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    out.extend_from_slice(&value.to_le_bytes());
    out.extend_from_slice(&unit.to_le_bytes());
    out
}

fn decode_quantity(name: &str, cell: &[u8]) -> Result<(f64, u16)> {
    if cell.len() != 10 {
        return Err(ZyronError::ExecutionError(format!(
            "{}: quantity value must be 10 bytes, got {}",
            name,
            cell.len()
        )));
    }
    let mut val = [0u8; 8];
    val.copy_from_slice(&cell[0..8]);
    let mut unit = [0u8; 2];
    unit.copy_from_slice(&cell[8..10]);
    Ok((f64::from_le_bytes(val), u16::from_le_bytes(unit)))
}

// CurrencyInfo has no Serialize derive so the composite cell is built as a
// JSON object by hand
fn currency_info_json(info: &zyron_types::money::CurrencyInfo) -> Vec<u8> {
    serde_json::json!({
        "code": info.code,
        "symbol": info.symbol,
        "decimals": info.decimals,
        "numeric": info.numeric,
    })
    .to_string()
    .into_bytes()
}

// ---------------------------------------------------------------------------
// argument readers
// ---------------------------------------------------------------------------

// broadcasts length-1 literal columns across the batch
fn bidx(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

// result row count for broadcast args, every length must equal the max or be
// 1, an empty column short-circuits to an empty result
fn out_len(name: &str, lens: &[usize]) -> Result<usize> {
    if lens.iter().any(|&l| l == 0) {
        return Ok(0);
    }
    let n = lens.iter().copied().max().unwrap_or(0);
    for &l in lens {
        if l != n && l != 1 {
            return Err(ZyronError::ExecutionError(format!(
                "{}: argument column length mismatch, {} vs {}",
                name, l, n
            )));
        }
    }
    Ok(n)
}

fn strings_arg<'a>(name: &str, col: &'a Column) -> Result<Vec<&'a str>> {
    super::column_strings(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected a string argument", name)))
}

fn floats_arg(name: &str, col: &Column) -> Result<Vec<f64>> {
    super::column_floats(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected a numeric argument", name)))
}

fn ints_arg(name: &str, col: &Column) -> Result<Vec<i64>> {
    super::column_ints(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected an integer argument", name)))
}

fn binary_cells<'a>(name: &str, col: &'a Column) -> Result<&'a Vec<Vec<u8>>> {
    match &col.data {
        ColumnData::Binary(v) => Ok(v),
        _ => Err(ZyronError::ExecutionError(format!(
            "{}: expected a binary encoded argument",
            name
        ))),
    }
}

// reads one currency numeric code per row, accepts a money binary cell, an
// alpha code string, or an integer numeric code, null rows yield a 0
// placeholder the caller masks out
fn currency_codes(name: &str, col: &Column) -> Result<Vec<u16>> {
    match &col.data {
        ColumnData::Binary(cells) => {
            let mut out = Vec::with_capacity(cells.len());
            for (i, cell) in cells.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(0);
                } else {
                    out.push(decode_money(name, cell)?.1);
                }
            }
            Ok(out)
        }
        ColumnData::Utf8(codes) => {
            let mut out = Vec::with_capacity(codes.len());
            for (i, code) in codes.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(0);
                    continue;
                }
                let info = zyron_types::money::currency_lookup(code).ok_or_else(|| {
                    ZyronError::ExecutionError(format!("{}: unknown currency {}", name, code))
                })?;
                out.push(info.numeric);
            }
            Ok(out)
        }
        _ => {
            let ints = ints_arg(name, col)?;
            Ok(ints.iter().map(|&v| v as u16).collect())
        }
    }
}

// reads one unit id per row, accepts a quantity binary cell, a unit name or
// symbol string, or an integer unit id, null rows yield a 0 placeholder the
// caller masks out
fn unit_ids(name: &str, col: &Column) -> Result<Vec<u16>> {
    match &col.data {
        ColumnData::Binary(cells) => {
            let mut out = Vec::with_capacity(cells.len());
            for (i, cell) in cells.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(0);
                } else {
                    out.push(decode_quantity(name, cell)?.1);
                }
            }
            Ok(out)
        }
        ColumnData::Utf8(names) => {
            let mut out = Vec::with_capacity(names.len());
            for (i, unit_name) in names.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(0);
                    continue;
                }
                let info = zyron_types::quantity::unit_lookup(unit_name).ok_or_else(|| {
                    ZyronError::ExecutionError(format!("{}: unknown unit {}", name, unit_name))
                })?;
                out.push(info.id);
            }
            Ok(out)
        }
        _ => {
            let ints = ints_arg(name, col)?;
            Ok(ints.iter().map(|&v| v as u16).collect())
        }
    }
}

fn arity_err(sig: &str, arity: &str) -> ZyronError {
    ZyronError::ExecutionError(format!("{} takes {}", sig, arity))
}

// ---------------------------------------------------------------------------
// money
// ---------------------------------------------------------------------------

// lookup miss returns NULL, not an error, matching validator style arms
fn currency_lookup_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(arity_err("currency_lookup(code)", "exactly 1 argument"));
    }
    let codes = strings_arg("currency_lookup", &args[0])?;
    let n = codes.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, code) in codes.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        match zyron_types::money::currency_lookup(code) {
            Some(info) => data.push(currency_info_json(&info)),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

// lookup miss and out-of-range numeric both return NULL
fn currency_by_numeric_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(arity_err(
            "currency_by_numeric(numeric)",
            "exactly 1 argument",
        ));
    }
    let nums = ints_arg("currency_by_numeric", &args[0])?;
    let n = nums.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, &num) in nums.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let info = u16::try_from(num)
            .ok()
            .and_then(zyron_types::money::currency_by_numeric);
        match info {
            Some(info) => data.push(currency_info_json(&info)),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

// unknown currency propagates as an error matching the module contract
fn money_create_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(
            "money_create(amount, currency)",
            "exactly 2 arguments",
        ));
    }
    let amounts = floats_arg("money_create", &args[0])?;
    let currencies = strings_arg("money_create", &args[1])?;
    let n = out_len("money_create", &[amounts.len(), currencies.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ai = bidx(amounts.len(), i);
        let ci = bidx(currencies.len(), i);
        if args[0].nulls.is_null(ai) || args[1].nulls.is_null(ci) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (val, cur) = zyron_types::money::money_create(amounts[ai], currencies[ci])?;
        data.push(encode_money(val, cur));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

// currency mismatch and overflow propagate as errors, the module exists to
// surface mixed-currency arithmetic bugs
fn money_binop_impl(
    name: &str,
    sig: &str,
    args: &[Column],
    op: fn(i64, u16, i64, u16) -> Result<(i64, u16)>,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(sig, "exactly 2 arguments"));
    }
    let a = binary_cells(name, &args[0])?;
    let b = binary_cells(name, &args[1])?;
    let n = out_len(name, &[a.len(), b.len()])?;
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
        let (av, ac) = decode_money(name, &a[ai])?;
        let (bv, bc) = decode_money(name, &b[bi])?;
        let (rv, rc) = op(av, ac, bv, bc)?;
        data.push(encode_money(rv, rc));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

fn money_multiply_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(
            "money_multiply(money, factor)",
            "exactly 2 arguments",
        ));
    }
    let cells = binary_cells("money_multiply", &args[0])?;
    let factors = floats_arg("money_multiply", &args[1])?;
    let n = out_len("money_multiply", &[cells.len(), factors.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let fi = bidx(factors.len(), i);
        if args[0].nulls.is_null(ci) || args[1].nulls.is_null(fi) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (val, cur) = decode_money("money_multiply", &cells[ci])?;
        let (rv, rc) = zyron_types::money::money_multiply(val, cur, factors[fi])?;
        data.push(encode_money(rv, rc));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

// target currency accepts an alpha code string, a numeric code integer, or
// another money value whose currency is used
fn money_convert_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 3 {
        return Err(arity_err(
            "money_convert(money, currency, rate)",
            "exactly 3 arguments",
        ));
    }
    let cells = binary_cells("money_convert", &args[0])?;
    let targets = currency_codes("money_convert", &args[1])?;
    let rates = floats_arg("money_convert", &args[2])?;
    let n = out_len("money_convert", &[cells.len(), targets.len(), rates.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let ti = bidx(targets.len(), i);
        let ri = bidx(rates.len(), i);
        if args[0].nulls.is_null(ci) || args[1].nulls.is_null(ti) || args[2].nulls.is_null(ri) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (val, cur) = decode_money("money_convert", &cells[ci])?;
        let (rv, rc) = zyron_types::money::money_convert(val, cur, targets[ti], rates[ri])?;
        data.push(encode_money(rv, rc));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

fn money_format_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(arity_err("money_format(money)", "exactly 1 argument"));
    }
    let cells = binary_cells("money_format", &args[0])?;
    let n = cells.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, cell) in cells.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let (val, cur) = decode_money("money_format", cell)?;
        data.push(zyron_types::money::money_format(val, cur));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// shared arm for money_currency_code and money_currency_symbol, the argument
// may be a money value or a numeric currency code
fn currency_str_impl<F: Fn(u16) -> String>(name: &str, args: &[Column], f: F) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{}(money) takes exactly 1 argument",
            name
        )));
    }
    let codes = currency_codes(name, &args[0])?;
    let n = codes.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, &code) in codes.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        data.push(f(code));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

fn money_minor_digits_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(arity_err("money_minor_digits(money)", "exactly 1 argument"));
    }
    let codes = currency_codes("money_minor_digits", &args[0])?;
    let n = codes.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, &code) in codes.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        data.push(zyron_types::money::money_minor_digits(code) as i32);
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

// ---------------------------------------------------------------------------
// quantity
// ---------------------------------------------------------------------------

// unknown unit propagates as an error matching the module contract
fn quantity_create_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(
            "quantity_create(value, unit)",
            "exactly 2 arguments",
        ));
    }
    let values = floats_arg("quantity_create", &args[0])?;
    let units = strings_arg("quantity_create", &args[1])?;
    let n = out_len("quantity_create", &[values.len(), units.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let vi = bidx(values.len(), i);
        let ui = bidx(units.len(), i);
        if args[0].nulls.is_null(vi) || args[1].nulls.is_null(ui) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (val, unit) = zyron_types::quantity::quantity_create(values[vi], units[ui])?;
        data.push(encode_quantity(val, unit));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Quantity,
    ))
}

// dimension mismatch propagates as an error, the module exists to surface
// unit mismatch bugs
fn quantity_binop_impl(
    name: &str,
    sig: &str,
    args: &[Column],
    op: fn(f64, u16, f64, u16) -> Result<(f64, u16)>,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(sig, "exactly 2 arguments"));
    }
    let a = binary_cells(name, &args[0])?;
    let b = binary_cells(name, &args[1])?;
    let n = out_len(name, &[a.len(), b.len()])?;
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
        let (av, au) = decode_quantity(name, &a[ai])?;
        let (bv, bu) = decode_quantity(name, &b[bi])?;
        let (rv, ru) = op(av, au, bv, bu)?;
        data.push(encode_quantity(rv, ru));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Quantity,
    ))
}

fn quantity_scale_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(
            "quantity_scale(quantity, factor)",
            "exactly 2 arguments",
        ));
    }
    let cells = binary_cells("quantity_scale", &args[0])?;
    let factors = floats_arg("quantity_scale", &args[1])?;
    let n = out_len("quantity_scale", &[cells.len(), factors.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let fi = bidx(factors.len(), i);
        if args[0].nulls.is_null(ci) || args[1].nulls.is_null(fi) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let (val, unit) = decode_quantity("quantity_scale", &cells[ci])?;
        let (rv, ru) = zyron_types::quantity::quantity_scale(val, unit, factors[fi]);
        data.push(encode_quantity(rv, ru));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Quantity,
    ))
}

// target unit accepts a unit name or symbol string, an integer unit id, or
// another quantity value whose unit is used
fn quantity_convert_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(
            "quantity_convert(quantity, unit)",
            "exactly 2 arguments",
        ));
    }
    let cells = binary_cells("quantity_convert", &args[0])?;
    let targets = unit_ids("quantity_convert", &args[1])?;
    let n = out_len("quantity_convert", &[cells.len(), targets.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let ti = bidx(targets.len(), i);
        if args[0].nulls.is_null(ci) || args[1].nulls.is_null(ti) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let (val, unit) = decode_quantity("quantity_convert", &cells[ci])?;
        data.push(zyron_types::quantity::quantity_convert(
            val,
            unit,
            targets[ti],
        )?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn quantity_format_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(arity_err("quantity_format(quantity)", "exactly 1 argument"));
    }
    let cells = binary_cells("quantity_format", &args[0])?;
    let n = cells.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, cell) in cells.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let (val, unit) = decode_quantity("quantity_format", cell)?;
        data.push(zyron_types::quantity::quantity_format(val, unit));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// shared arm for quantity_dimension and quantity_unit_name, the argument may
// be a quantity value, a unit id, or a unit name
fn unit_str_impl<F: Fn(u16) -> String>(name: &str, args: &[Column], f: F) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{}(quantity) takes exactly 1 argument",
            name
        )));
    }
    let units = unit_ids(name, &args[0])?;
    let n = units.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, &unit) in units.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        data.push(f(unit));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

// unknown unit or dimension mismatch propagates as an error
fn convert_units_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 3 {
        return Err(arity_err(
            "convert_units(value, from_unit, to_unit)",
            "exactly 3 arguments",
        ));
    }
    let values = floats_arg("convert_units", &args[0])?;
    let from = strings_arg("convert_units", &args[1])?;
    let to = strings_arg("convert_units", &args[2])?;
    let n = out_len("convert_units", &[values.len(), from.len(), to.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let vi = bidx(values.len(), i);
        let fi = bidx(from.len(), i);
        let ti = bidx(to.len(), i);
        if args[0].nulls.is_null(vi) || args[1].nulls.is_null(fi) || args[2].nulls.is_null(ti) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        data.push(zyron_types::formatting::convert_units(
            values[vi], from[fi], to[ti],
        )?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// locale is optional and defaults to en
fn format_currency_impl(args: &[Column]) -> Result<Column> {
    if args.len() < 2 || args.len() > 3 {
        return Err(arity_err(
            "format_currency(value, currency [, locale])",
            "2 or 3 arguments",
        ));
    }
    let values = floats_arg("format_currency", &args[0])?;
    let currencies = strings_arg("format_currency", &args[1])?;
    let locales = if args.len() == 3 {
        Some(strings_arg("format_currency", &args[2])?)
    } else {
        None
    };
    let n = out_len(
        "format_currency",
        &[
            values.len(),
            currencies.len(),
            locales.as_ref().map_or(1, |l| l.len()),
        ],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let vi = bidx(values.len(), i);
        let ci = bidx(currencies.len(), i);
        let locale_null = locales
            .as_ref()
            .is_some_and(|l| args[2].nulls.is_null(bidx(l.len(), i)));
        if args[0].nulls.is_null(vi) || args[1].nulls.is_null(ci) || locale_null {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let locale = locales.as_ref().map_or("en", |l| l[bidx(l.len(), i)]);
        data.push(zyron_types::formatting::format_currency(
            values[vi],
            currencies[ci],
            locale,
        ));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// locale is optional and defaults to en
fn format_number_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err(
            "format_number(value [, locale])",
            "1 or 2 arguments",
        ));
    }
    let values = floats_arg("format_number", &args[0])?;
    let locales = if args.len() == 2 {
        Some(strings_arg("format_number", &args[1])?)
    } else {
        None
    };
    let n = out_len(
        "format_number",
        &[values.len(), locales.as_ref().map_or(1, |l| l.len())],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let vi = bidx(values.len(), i);
        let locale_null = locales
            .as_ref()
            .is_some_and(|l| args[1].nulls.is_null(bidx(l.len(), i)));
        if args[0].nulls.is_null(vi) || locale_null {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let locale = locales.as_ref().map_or("en", |l| l[bidx(l.len(), i)]);
        data.push(zyron_types::formatting::format_number(values[vi], locale));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// decimals is optional and defaults to 2, clamped to 0..=32 to bound output
fn format_percentage_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err(
            "format_percentage(value [, decimals])",
            "1 or 2 arguments",
        ));
    }
    let values = floats_arg("format_percentage", &args[0])?;
    let decimals = if args.len() == 2 {
        Some(ints_arg("format_percentage", &args[1])?)
    } else {
        None
    };
    let n = out_len(
        "format_percentage",
        &[values.len(), decimals.as_ref().map_or(1, |d| d.len())],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let vi = bidx(values.len(), i);
        let dec_null = decimals
            .as_ref()
            .is_some_and(|d| args[1].nulls.is_null(bidx(d.len(), i)));
        if args[0].nulls.is_null(vi) || dec_null {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let dec = decimals
            .as_ref()
            .map_or(2, |d| d[bidx(d.len(), i)].clamp(0, 32)) as u32;
        data.push(zyron_types::formatting::format_percentage(values[vi], dec));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// parser style arm, a row that fails to parse yields NULL instead of failing
// the batch, locale is optional and defaults to en
fn parse_number_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err(
            "parse_number(text [, locale])",
            "1 or 2 arguments",
        ));
    }
    let texts = strings_arg("parse_number", &args[0])?;
    let locales = if args.len() == 2 {
        Some(strings_arg("parse_number", &args[1])?)
    } else {
        None
    };
    let n = out_len(
        "parse_number",
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
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let locale = locales.as_ref().map_or("en", |l| l[bidx(l.len(), i)]);
        match zyron_types::formatting::parse_number(texts[ti], locale) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(values: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(values.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn f64_col(values: &[f64]) -> Column {
        Column::new(ColumnData::Float64(values.to_vec()), TypeId::Float64)
    }

    #[test]
    fn money_create_and_format_roundtrip() {
        let money = dispatch("money_create", &[f64_col(&[19.99]), utf8_col(&["USD"])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(money.type_id, TypeId::Money);
        let formatted = dispatch("money_format", &[money], 1).unwrap().unwrap();
        match &formatted.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "$19.99"),
            other => panic!("expected Utf8, got {:?}", other),
        }
    }

    #[test]
    fn money_add_same_currency() {
        let a = Column::new(
            ColumnData::Binary(vec![encode_money(1999, 840)]),
            TypeId::Money,
        );
        let b = Column::new(
            ColumnData::Binary(vec![encode_money(500, 840)]),
            TypeId::Money,
        );
        let out = dispatch("money_add", &[a, b], 1).unwrap().unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                let (val, cur) = decode_money("test", &v[0]).unwrap();
                assert_eq!(val, 2499);
                assert_eq!(cur, 840);
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn money_add_mismatched_currency_errors() {
        let a = Column::new(
            ColumnData::Binary(vec![encode_money(100, 840)]),
            TypeId::Money,
        );
        let b = Column::new(
            ColumnData::Binary(vec![encode_money(100, 978)]),
            TypeId::Money,
        );
        assert!(dispatch("money_add", &[a, b], 1).unwrap().is_err());
    }

    #[test]
    fn money_format_propagates_null() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(0);
        let col = Column::with_nulls(
            ColumnData::Binary(vec![Vec::new(), encode_money(1999, 840)]),
            nulls,
            TypeId::Money,
        );
        let out = dispatch("money_format", &[col], 2).unwrap().unwrap();
        assert!(out.nulls.is_null(0));
        assert!(!out.nulls.is_null(1));
        match &out.data {
            ColumnData::Utf8(v) => assert_eq!(v[1], "$19.99"),
            other => panic!("expected Utf8, got {:?}", other),
        }
    }

    #[test]
    fn quantity_add_converts_across_units() {
        // 1 kg (unit 20) + 500 g (unit 21) = 1.5 kg
        let a = Column::new(
            ColumnData::Binary(vec![encode_quantity(1.0, 20)]),
            TypeId::Quantity,
        );
        let b = Column::new(
            ColumnData::Binary(vec![encode_quantity(500.0, 21)]),
            TypeId::Quantity,
        );
        let out = dispatch("quantity_add", &[a, b], 1).unwrap().unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                let (val, unit) = decode_quantity("test", &v[0]).unwrap();
                assert!((val - 1.5).abs() < 1e-9);
                assert_eq!(unit, 20);
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn quantity_convert_accepts_text_target() {
        // 1 km (unit 2) to miles
        let q = Column::new(
            ColumnData::Binary(vec![encode_quantity(1.0, 2)]),
            TypeId::Quantity,
        );
        let out = dispatch("quantity_convert", &[q, utf8_col(&["mile"])], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Float64(v) => assert!((v[0] - 0.621371).abs() < 0.001),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn parse_number_bad_row_yields_null() {
        let out = dispatch("parse_number", &[utf8_col(&["1,234.56", "junk"])], 2)
            .unwrap()
            .unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::Float64(v) => assert!((v[0] - 1234.56).abs() < 1e-9),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn currency_lookup_unknown_yields_null() {
        let out = dispatch("currency_lookup", &[utf8_col(&["USD", "XYZ"])], 2)
            .unwrap()
            .unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::Binary(v) => {
                let json = String::from_utf8(v[0].clone()).unwrap();
                assert!(json.contains("\"code\":\"USD\""));
                assert!(json.contains("\"numeric\":840"));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }
}
