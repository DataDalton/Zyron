//! Dispatch arms for the zyron_sys.security.masking_* PII anonymization
//! family. Every function takes text in and returns text, with per row
//! null passthrough

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "masking_ip" => masking_ip_impl(args, num_rows),
        "masking_email" => text_unary_impl(
            "masking_email(email)",
            args,
            num_rows,
            zyron_types::masking::masking_email,
        ),
        "masking_phone" => masking_phone_impl(args, num_rows),
        "masking_ssn" => text_unary_impl(
            "masking_ssn(ssn)",
            args,
            num_rows,
            zyron_types::masking::masking_ssn,
        ),
        "masking_name" => text_unary_impl(
            "masking_name(name)",
            args,
            num_rows,
            zyron_types::masking::masking_name,
        ),
        _ => return None,
    })
}

fn text_rows(sig: &str, col: &Column) -> Result<Vec<String>> {
    super::column_strings(col)
        .map(|v| v.into_iter().map(str::to_string).collect())
        .map_err(|_| ZyronError::ExecutionError(format!("{sig}: expected a string argument")))
}

fn build_text_output(
    input: &Column,
    num_rows: usize,
    mut per_row: impl FnMut(usize) -> Result<String>,
) -> Result<Column> {
    let rows = num_rows.max(1);
    let mut out = Vec::with_capacity(rows);
    let mut nulls = NullBitmap::none(rows);
    for row in 0..rows {
        if input.nulls.is_null(row) {
            nulls.set_null(row);
            out.push(String::new());
            continue;
        }
        out.push(per_row(row)?);
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

fn text_unary_impl(
    sig: &str,
    args: &[Column],
    num_rows: usize,
    f: fn(&str) -> Result<String>,
) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{sig} takes exactly one argument"
        )));
    }
    let values = text_rows(sig, &args[0])?;
    build_text_output(&args[0], num_rows, |row| match values.get(row) {
        Some(v) => f(v),
        None => Err(ZyronError::ExecutionError(format!(
            "{sig}: row out of range"
        ))),
    })
}

fn masking_ip_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "masking_ip(ip, keep_prefix_bits)";
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes an address and an optional prefix length"
        )));
    }
    let values = text_rows(SIG, &args[0])?;
    let bits = match args.get(1) {
        Some(col) => super::column_ints(col)
            .map_err(|_| {
                ZyronError::ExecutionError(format!("{SIG}: keep_prefix_bits must be an integer"))
            })?
            .first()
            .copied()
            .unwrap_or(24),
        None => 24,
    };
    if bits < 0 {
        return Err(ZyronError::InvalidParameter {
            name: "keep_prefix_bits".to_string(),
            value: bits.to_string(),
        });
    }
    build_text_output(&args[0], num_rows, |row| {
        zyron_types::masking::masking_ip(&values[row], bits as u32)
    })
}

fn masking_phone_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "masking_phone(phone, keep_country_code)";
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes a number and an optional keep_country_code flag"
        )));
    }
    let values = text_rows(SIG, &args[0])?;
    let keep = match args.get(1) {
        Some(col) => match &col.data {
            ColumnData::Boolean(v) => v.first().copied().unwrap_or(true),
            _ => {
                return Err(ZyronError::ExecutionError(format!(
                    "{SIG}: keep_country_code must be a boolean"
                )));
            }
        },
        None => true,
    };
    build_text_output(&args[0], num_rows, |row| {
        zyron_types::masking::masking_phone(&values[row], keep)
    })
}
