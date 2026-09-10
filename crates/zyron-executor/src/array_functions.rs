//! The array functions that let unnested rows go back into arrays.
//!
//! Every one of them reads the encoded array in place through `ArrayView`
//! and writes a new encoded array, so a value is never materialized into a
//! list of scalars just to be re-encoded.
//!
//! The two higher-order forms, `array_filter` and `array_transform`, do not
//! evaluate their lambda once per element. Every element of every row in the
//! batch is gathered into one column, the body is evaluated over that column
//! in a single pass, and the results are cut back into arrays by the lengths
//! the rows had. Ten million rows of ten elements is therefore one expression
//! evaluation over a hundred million values, not a hundred million
//! evaluations.

use zyron_common::{ArrayView, Result, TypeId, ZyronError, array_value};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::{LAMBDA_TABLE_IDX, LogicalColumn};

use crate::batch::{ColumnBuilder, DataBatch, decode_fixed_scalar, decode_varlen_scalar};
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};

/// True when the name is one of the array functions this module evaluates.
pub fn is_array_function(name: &str) -> bool {
    matches!(
        name,
        "array_length"
            | "array_position"
            | "array_contains"
            | "array_distinct"
            | "array_sort"
            | "array_slice"
            | "array_concat"
            | "array_to_string"
            | "string_to_array"
            | "array_filter"
            | "array_transform"
    )
}

/// Evaluates one array function over a batch.
pub fn evaluate_array_function(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let eval = |expr: &BoundExpr| crate::expr::evaluate(expr, batch, schema, params);
    match name {
        "array_length" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let rows = arrays.len();
            let mut builder = ColumnBuilder::new(TypeId::Int64, rows);
            for row in 0..rows {
                builder.push(&match view_at(&arrays, row) {
                    Some(view) => ScalarValue::Int64(view.len() as i64),
                    None => ScalarValue::Null,
                });
            }
            Ok(builder.finish())
        }
        "array_position" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let needles = eval(arg(name, args, 1)?)?;
            let rows = arrays.len();
            let mut builder = ColumnBuilder::new(TypeId::Int64, rows);
            // A needle given as a literal is one value for the whole batch,
            // so it is read once rather than cloned out of the column per row
            let single_needle = (needles.len() == 1).then(|| needles.get_scalar(0));
            for row in 0..rows {
                let found = match view_at(&arrays, row) {
                    None => None,
                    Some(view) => match &single_needle {
                        Some(needle) => position_of(&view, needle),
                        None => {
                            let needle =
                                needles.get_scalar(row.min(needles.len().saturating_sub(1)));
                            position_of(&view, &needle)
                        }
                    },
                };
                builder.push(&match found {
                    Some(at) => ScalarValue::Int64(at as i64 + 1),
                    None => ScalarValue::Null,
                });
            }
            Ok(builder.finish())
        }
        "array_contains" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let needles = eval(arg(name, args, 1)?)?;
            let rows = arrays.len();
            let mut builder = ColumnBuilder::new(TypeId::Boolean, rows);
            // Read once for the whole batch when the needle is a literal
            let single_needle = (needles.len() == 1).then(|| needles.get_scalar(0));
            for row in 0..rows {
                builder.push(&match view_at(&arrays, row) {
                    None => ScalarValue::Null,
                    Some(view) => match &single_needle {
                        Some(needle) => ScalarValue::Boolean(position_of(&view, needle).is_some()),
                        None => {
                            let needle =
                                needles.get_scalar(row.min(needles.len().saturating_sub(1)));
                            ScalarValue::Boolean(position_of(&view, &needle).is_some())
                        }
                    },
                });
            }
            Ok(builder.finish())
        }
        "array_distinct" => {
            let arrays = eval(arg(name, args, 0)?)?;
            rebuild_arrays(&arrays, distinct_elements)
        }
        "array_sort" => {
            let arrays = eval(arg(name, args, 0)?)?;
            rebuild_arrays(&arrays, sort_elements)
        }
        "array_slice" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let starts = eval(arg(name, args, 1)?)?;
            let lengths = eval(arg(name, args, 2)?)?;
            let rows = arrays.len();
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(rows);
            let mut nulls = NullBitmap::empty();
            let mut kept: Vec<Option<&[u8]>> = Vec::new();
            for row in 0..rows {
                match view_at(&arrays, row) {
                    None => {
                        nulls.push(true);
                        out.push(Vec::new());
                    }
                    Some(view) => {
                        let start = as_i64(&starts.get_scalar(row)).unwrap_or(1).max(1) as usize;
                        let take = as_i64(&lengths.get_scalar(row)).unwrap_or(0).max(0) as usize;
                        // The surviving elements are named by slices into
                        // the row's payload, so the slice copies each one
                        // into the encoded result and nowhere else
                        kept.clear();
                        kept.extend(
                            (start - 1..view.len())
                                .take(take)
                                .map(|i| view.get(i).flatten()),
                        );
                        nulls.push(false);
                        out.push(array_value::encode(view.element_type(), &kept));
                    }
                }
            }
            Ok(Column::with_nulls(
                ColumnData::Binary(out),
                nulls,
                TypeId::Array,
            ))
        }
        "array_concat" => {
            if args.len() < 2 {
                return Err(ZyronError::ExecutionError(
                    "array_concat takes at least two arrays".to_string(),
                ));
            }
            let columns: Vec<Column> = args.iter().map(eval).collect::<Result<Vec<_>>>()?;
            let rows = columns.iter().map(|c| c.len()).max().unwrap_or(0);
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(rows);
            let mut nulls = NullBitmap::empty();
            // One buffer of borrows for every row, rather than a vector of
            // copied elements allocated per row
            let mut joined: Vec<Option<&[u8]>> = Vec::new();
            for row in 0..rows {
                let mut element_type = TypeId::Null;
                joined.clear();
                let mut any = false;
                for column in &columns {
                    if let Some(view) = view_at(column, row.min(column.len().saturating_sub(1))) {
                        any = true;
                        if element_type == TypeId::Null {
                            element_type = view.element_type();
                        }
                        joined.extend(view.iter());
                    }
                }
                if any {
                    nulls.push(false);
                    out.push(array_value::encode(element_type, &joined));
                } else {
                    nulls.push(true);
                    out.push(Vec::new());
                }
            }
            Ok(Column::with_nulls(
                ColumnData::Binary(out),
                nulls,
                TypeId::Array,
            ))
        }
        "array_to_string" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let delimiters = eval(arg(name, args, 1)?)?;
            // The third argument is what a null element renders as. Without
            // it a null element is skipped, which is what the SQL form does
            let null_text = match args.get(2) {
                Some(expr) => Some(eval(expr)?),
                None => None,
            };
            let rows = arrays.len();
            let mut builder = ColumnBuilder::new(TypeId::Varchar, rows);
            // The delimiter and the null replacement are almost always
            // literals, so each is rendered once for the batch. Rendering
            // them per row put two string allocations on every row for a
            // value that never changed
            let mut delimiter = String::new();
            let mut replacement = String::new();
            let mut text = String::new();
            for row in 0..rows {
                match view_at(&arrays, row) {
                    None => builder.push(&ScalarValue::Null),
                    Some(view) => {
                        render_scalar_into(
                            &mut delimiter,
                            &delimiters.get_scalar(row.min(delimiters.len().saturating_sub(1))),
                        );
                        let has_replacement = match &null_text {
                            Some(c) => {
                                render_scalar_into(
                                    &mut replacement,
                                    &c.get_scalar(row.min(c.len().saturating_sub(1))),
                                );
                                true
                            }
                            None => false,
                        };
                        text.clear();
                        for element in view.iter() {
                            match element {
                                Some(bytes) => {
                                    if !text.is_empty() {
                                        text.push_str(&delimiter);
                                    }
                                    // Rendered straight into the accumulator,
                                    // so an element costs no string of its own
                                    render_element_into(&mut text, view.element_type(), bytes);
                                }
                                None => {
                                    if !has_replacement {
                                        continue;
                                    }
                                    if !text.is_empty() {
                                        text.push_str(&delimiter);
                                    }
                                    text.push_str(&replacement);
                                }
                            }
                        }
                        builder.push_owned(ScalarValue::Utf8(std::mem::take(&mut text)));
                    }
                }
            }
            Ok(builder.finish())
        }
        "string_to_array" => {
            let texts = eval(arg(name, args, 0)?)?;
            let delimiters = eval(arg(name, args, 1)?)?;
            // The third argument is the text that reads back as a null
            // element rather than as itself
            let null_text = match args.get(2) {
                Some(expr) => Some(eval(expr)?),
                None => None,
            };
            let rows = texts.len();
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(rows);
            let mut nulls = NullBitmap::empty();
            // Rendered into buffers the rows share rather than into three
            // fresh strings per row, two of which hold a literal
            let mut text = String::new();
            let mut delimiter = String::new();
            let mut marker = String::new();
            for row in 0..rows {
                if texts.is_null(row) {
                    nulls.push(true);
                    out.push(Vec::new());
                    continue;
                }
                render_scalar_into(&mut text, &texts.get_scalar(row));
                render_scalar_into(
                    &mut delimiter,
                    &delimiters.get_scalar(row.min(delimiters.len().saturating_sub(1))),
                );
                let has_marker = match &null_text {
                    Some(c) => {
                        render_scalar_into(
                            &mut marker,
                            &c.get_scalar(row.min(c.len().saturating_sub(1))),
                        );
                        true
                    }
                    None => false,
                };
                // Scoped to the row because its slices borrow the text
                // buffer the next row overwrites. One vector per row, and
                // none per part, which is where the cost was
                let mut parts: Vec<Option<&[u8]>> = Vec::new();
                if delimiter.is_empty() {
                    // An empty delimiter splits nothing, so the whole text is
                    // one element
                    parts.push(Some(text.as_bytes()));
                } else {
                    parts.extend(text.split(delimiter.as_str()).map(|part| {
                        if has_marker && part == marker {
                            None
                        } else {
                            Some(part.as_bytes())
                        }
                    }))
                };
                nulls.push(false);
                out.push(array_value::encode(TypeId::Text, &parts));
            }
            Ok(Column::with_nulls(
                ColumnData::Binary(out),
                nulls,
                TypeId::Array,
            ))
        }
        "array_filter" | "array_transform" => {
            let arrays = eval(arg(name, args, 0)?)?;
            let body = arg(name, args, 1)?;
            evaluate_higher_order(name, &arrays, body, params)
        }
        other => Err(ZyronError::ExecutionError(format!(
            "array function '{other}' has no evaluator"
        ))),
    }
}

/// Evaluates a lambda body over every element of every row at once, then cuts
/// the results back into one array per row.
fn evaluate_higher_order(
    name: &str,
    arrays: &Column,
    body: &BoundExpr,
    params: &[ScalarValue],
) -> Result<Column> {
    let rows = arrays.len();
    // The element type comes from the encoded value, so a column whose
    // element type the plan did not know still evaluates
    let element_type = (0..rows)
        .find_map(|row| view_at(arrays, row).map(|v| v.element_type()))
        .unwrap_or(TypeId::Null);

    // A filter hands its elements back unchanged, so their stored bytes are
    // kept to be written out again. A transform replaces every element, so
    // keeping them would be an allocation per element for a value nothing
    // reads
    let filtering = name == "array_filter";

    // Every element of every row, flattened, with where each row's run starts
    let mut flat = ColumnBuilder::new(element_type, rows * 4);
    let mut starts: Vec<usize> = Vec::with_capacity(rows + 1);
    // The elements a filter keeps travel unchanged, so they are named by
    // slices into the column they came from rather than copied out of it.
    // Every row's view borrows the same column, so one vector of borrows
    // serves the whole batch
    let mut originals: Vec<Option<&[u8]>> = Vec::new();
    let mut present: Vec<bool> = Vec::with_capacity(rows);
    let mut total = 0usize;
    for row in 0..rows {
        starts.push(total);
        match view_at(arrays, row) {
            None => present.push(false),
            Some(view) => {
                present.push(true);
                for element in view.iter() {
                    flat.push(&match element {
                        Some(bytes) => decode_element(element_type, bytes),
                        None => ScalarValue::Null,
                    });
                    if filtering {
                        originals.push(element);
                    }
                    total += 1;
                }
            }
        }
    }
    starts.push(total);

    // One evaluation over the whole flattened column
    let element_column = flat.finish();
    let element_schema = vec![LogicalColumn {
        table_idx: Some(LAMBDA_TABLE_IDX),
        column_id: zyron_catalog::ColumnId(0),
        name: "element".to_string(),
        type_id: element_type,
        nullable: true,
        fractional_digits: None,
    }];
    let element_batch = DataBatch::new(vec![element_column]);
    let applied = crate::expr::evaluate(body, &element_batch, &element_schema, params)?;

    let result_type = if name == "array_filter" {
        element_type
    } else {
        applied.type_id
    };
    let result_width = result_type.fixed_size().unwrap_or(0);
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(rows);
    let mut nulls = NullBitmap::empty();
    // One buffer for every element's bytes, reused across rows. Encoding
    // into a fresh Vec per element is an allocation per element, which for
    // ten million rows of ten elements is a hundred million of them
    let mut payload = Vec::with_capacity(result_width.max(16));
    // The element payloads of one row, as ranges into `payload` plus a null
    // flag, so the encoder borrows them without a copy each
    let mut spans: Vec<Option<(usize, usize)>> = Vec::new();
    for row in 0..rows {
        if !present[row] {
            nulls.push(true);
            out.push(Vec::new());
            continue;
        }
        payload.clear();
        spans.clear();
        for at in starts[row]..starts[row + 1] {
            if filtering {
                // The element travels unchanged, so its stored bytes are
                // reused rather than decoded and re-encoded
                if matches!(applied.get_scalar(at), ScalarValue::Boolean(true)) {
                    match originals[at] {
                        Some(bytes) => {
                            let from = payload.len();
                            payload.extend_from_slice(bytes);
                            spans.push(Some((from, payload.len())));
                        }
                        None => spans.push(None),
                    }
                }
                continue;
            }
            if applied.is_null(at) {
                spans.push(None);
                continue;
            }
            let from = payload.len();
            // A text or binary result is already bytes in the produced
            // column, so it is copied straight into the payload. Reading it
            // as a scalar first would allocate a String or a Vec per element
            // only to copy out of it and drop it
            match &applied.data {
                ColumnData::Utf8(values) => payload.extend_from_slice(values[at].as_bytes()),
                ColumnData::Binary(values) => payload.extend_from_slice(&values[at]),
                _ => crate::batch::encode_scalar_value_into(
                    &mut payload,
                    result_type,
                    &applied.get_scalar(at),
                    result_width,
                ),
            }
            spans.push(Some((from, payload.len())));
        }
        nulls.push(false);
        out.push(array_value::encode_spans(result_type, &payload, &spans));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(out),
        nulls,
        TypeId::Array,
    ))
}

/// One argument of an array function, or a clear error naming the function.
fn arg<'a>(name: &str, args: &'a [BoundExpr], at: usize) -> Result<&'a BoundExpr> {
    args.get(at).ok_or_else(|| {
        ZyronError::ExecutionError(format!("{name} takes at least {} argument(s)", at + 1))
    })
}

/// The encoded array one row holds, or None when the row is null or holds
/// something that is not an array.
fn view_at<'a>(column: &'a Column, row: usize) -> Option<ArrayView<'a>> {
    if row >= column.len() || column.is_null(row) {
        return None;
    }
    let ColumnData::Binary(payloads) = &column.data else {
        return None;
    };
    ArrayView::parse(payloads.get(row)?)
}

/// Rewrites every row's array through a transform over its elements.
fn rebuild_arrays<'a, F>(arrays: &'a Column, transform: F) -> Result<Column>
where
    F: Fn(&ArrayView<'a>, &mut Vec<Option<&'a [u8]>>),
{
    let rows = arrays.len();
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(rows);
    let mut nulls = NullBitmap::empty();
    // The elements a row keeps are named by slices into the column's own
    // payload, so a rebuild copies each surviving element once, into the
    // encoded result, rather than once into a vector and once again out of
    // it. Every row's view borrows the same column, so one buffer serves
    // every row rather than one being allocated per row
    let mut kept: Vec<Option<&'a [u8]>> = Vec::new();
    for row in 0..rows {
        match view_at(arrays, row) {
            None => {
                nulls.push(true);
                out.push(Vec::new());
            }
            Some(view) => {
                kept.clear();
                transform(&view, &mut kept);
                nulls.push(false);
                out.push(array_value::encode(view.element_type(), &kept));
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(out),
        nulls,
        TypeId::Array,
    ))
}

/// The zero-based position of the first element equal to a value.
///
/// Elements are decoded and compared as values rather than as bytes, because
/// an INT[] stores four-byte elements while a whole number literal binds to
/// eight, and comparing the bytes would answer that the value is absent.
fn position_of(view: &ArrayView<'_>, needle: &ScalarValue) -> Option<usize> {
    if matches!(needle, ScalarValue::Null) {
        // A null equals nothing, so it is never found
        return None;
    }
    let element_type = view.element_type();
    view.iter().position(|element| match element {
        Some(bytes) => {
            crate::correlated::scalar_eq(&decode_element(element_type, bytes), needle) == Some(true)
        }
        None => false,
    })
}

/// Decodes one element's payload into the value it holds.
fn decode_element(element_type: TypeId, bytes: &[u8]) -> ScalarValue {
    if element_type.fixed_size().unwrap_or(0) > 0 {
        decode_fixed_scalar(element_type, bytes)
    } else {
        decode_varlen_scalar(element_type, bytes)
    }
}

/// Keeps each distinct element, in the order the array first held it.
///
/// Membership is a hash of the element's bytes rather than a walk of what was
/// kept, so a long array costs one pass rather than a comparison against
/// every element already taken.
fn distinct_elements<'a>(view: &ArrayView<'a>, kept: &mut Vec<Option<&'a [u8]>>) {
    let mut seen: std::collections::HashSet<Option<&[u8]>> =
        std::collections::HashSet::with_capacity(view.len());
    for element in view.iter() {
        if seen.insert(element) {
            kept.push(element);
        }
    }
}

/// Orders an array's elements by the values they hold, nulls last.
///
/// Each element is decoded once and sorted by the decoded value. Decoding
/// both sides inside the comparator would decode every element as many times
/// as the sort compares it, which for a text element is an allocation per
/// comparison rather than one per element.
fn sort_elements<'a>(view: &ArrayView<'a>, kept: &mut Vec<Option<&'a [u8]>>) {
    let element_type = view.element_type();
    let mut decorated: Vec<(Option<ScalarValue>, Option<&'a [u8]>)> = view
        .iter()
        .map(|e| (e.map(|b| decode_element(element_type, b)), e))
        .collect();
    decorated.sort_by(|a, b| match (&a.0, &b.0) {
        // A null sorts last, which is where an ascending order puts an
        // absent value
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (Some(_), None) => std::cmp::Ordering::Less,
        (Some(x), Some(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
    });
    kept.extend(decorated.into_iter().map(|(_, bytes)| bytes));
}

/// Renders a value into a buffer the caller reuses.
///
/// The text forms write their bytes straight in, and the numeric forms go
/// through the formatter without a string of their own, so a value rendered
/// once per row costs the buffer's growth rather than an allocation per row.
fn render_scalar_into(out: &mut String, value: &ScalarValue) {
    use std::fmt::Write;
    out.clear();
    match value {
        ScalarValue::Utf8(s) => out.push_str(s),
        ScalarValue::Null => {}
        ScalarValue::Binary(b) => out.push_str(&String::from_utf8_lossy(b)),
        ScalarValue::Boolean(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Int8(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Int16(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Int32(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Int64(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Int128(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::UInt8(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::UInt16(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::UInt32(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::UInt64(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Float32(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::Float64(v) => {
            let _ = write!(out, "{v}");
        }
        ScalarValue::FixedBinary16(b) => {
            for byte in b {
                let _ = write!(out, "{byte:02x}");
            }
        }
        ScalarValue::Interval(i) => {
            let _ = write!(out, "{i:?}");
        }
    }
}

/// Appends one encoded element's rendering to a buffer.
///
/// A text element's bytes go straight in. Every other element is decoded and
/// rendered, which is the one allocation the decode itself owns.
fn render_element_into(out: &mut String, element_type: TypeId, bytes: &[u8]) {
    if matches!(element_type, TypeId::Text | TypeId::Varchar | TypeId::Char) {
        out.push_str(&String::from_utf8_lossy(bytes));
        return;
    }
    let mut rendered = String::new();
    render_scalar_into(&mut rendered, &decode_element(element_type, bytes));
    out.push_str(&rendered);
}

/// A value as a whole number, for the positional arguments a slice takes.
fn as_i64(value: &ScalarValue) -> Option<i64> {
    match value {
        ScalarValue::Int8(v) => Some(*v as i64),
        ScalarValue::Int16(v) => Some(*v as i64),
        ScalarValue::Int32(v) => Some(*v as i64),
        ScalarValue::Int64(v) => Some(*v),
        ScalarValue::UInt8(v) => Some(*v as i64),
        ScalarValue::UInt16(v) => Some(*v as i64),
        ScalarValue::UInt32(v) => Some(*v as i64),
        ScalarValue::UInt64(v) => Some(*v as i64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_array(values: &[Option<i64>]) -> Vec<u8> {
        let payloads: Vec<Option<[u8; 8]>> =
            values.iter().map(|v| v.map(i64::to_le_bytes)).collect();
        let borrowed: Vec<Option<&[u8]>> = payloads
            .iter()
            .map(|p| p.as_ref().map(|b| &b[..]))
            .collect();
        array_value::encode(TypeId::Int64, &borrowed)
    }

    /// Runs one element transform the way `rebuild_arrays` does.
    fn transformed<'a>(
        encoded: &'a [u8],
        f: impl Fn(&ArrayView<'a>, &mut Vec<Option<&'a [u8]>>),
    ) -> Vec<u8> {
        let view = ArrayView::parse(encoded).expect("parses");
        let mut kept: Vec<Option<&[u8]>> = Vec::new();
        f(&view, &mut kept);
        array_value::encode(view.element_type(), &kept)
    }

    fn read_ints(encoded: &[u8]) -> Vec<ScalarValue> {
        let view = ArrayView::parse(encoded).expect("parses");
        view.iter()
            .map(|e| match e {
                Some(b) => decode_element(TypeId::Int64, b),
                None => ScalarValue::Null,
            })
            .collect()
    }

    #[test]
    fn a_position_is_found_by_the_value_and_never_by_a_null() {
        let encoded = int_array(&[Some(10), None, Some(30)]);
        let view = ArrayView::parse(&encoded).expect("parses");
        assert_eq!(position_of(&view, &ScalarValue::Int64(30)), Some(2));
        assert_eq!(position_of(&view, &ScalarValue::Int64(99)), None);
        assert_eq!(position_of(&view, &ScalarValue::Null), None);
    }

    #[test]
    fn elements_order_by_their_values_with_nulls_last() {
        let sorted = transformed(
            &int_array(&[Some(3), None, Some(1), Some(2)]),
            sort_elements,
        );
        assert_eq!(
            read_ints(&sorted),
            vec![
                ScalarValue::Int64(1),
                ScalarValue::Int64(2),
                ScalarValue::Int64(3),
                ScalarValue::Null,
            ]
        );
    }

    #[test]
    fn distinct_keeps_the_first_appearance_of_each_element() {
        let kept = transformed(
            &int_array(&[Some(2), Some(1), Some(2), None, Some(1), None]),
            distinct_elements,
        );
        assert_eq!(
            read_ints(&kept),
            vec![
                ScalarValue::Int64(2),
                ScalarValue::Int64(1),
                ScalarValue::Null,
            ],
            "each distinct element is kept once, in first appearance order"
        );
    }

    #[test]
    fn an_empty_array_round_trips_through_a_rebuild() {
        let encoded = int_array(&[]);
        let view = ArrayView::parse(&encoded).expect("parses");
        assert_eq!(view.len(), 0);
        let rebuilt = transformed(&encoded, sort_elements);
        assert_eq!(ArrayView::parse(&rebuilt).map(|v| v.len()), Some(0));
    }

    #[test]
    fn every_name_this_module_answers_to_is_listed() {
        for name in [
            "array_length",
            "array_position",
            "array_contains",
            "array_distinct",
            "array_sort",
            "array_slice",
            "array_concat",
            "array_to_string",
            "string_to_array",
            "array_filter",
            "array_transform",
        ] {
            assert!(is_array_function(name), "{name} is not listed");
        }
        assert!(!is_array_function("array"));
        assert!(!is_array_function("array_subscript"));
    }
}
