//! Turns STRUCT and MAP values between the JSON a statement writes and the
//! binary a column stores.
//!
//! A declared shape is what makes the binary form worth having. The names sit
//! in the declaration, so the value holds only bytes, and a field is reached
//! by its position rather than by scanning for its name. Encoding happens
//! once on write; the read path addresses fields directly and only renders
//! back to JSON when a whole value leaves the engine.
//!
//! A leaf field's bytes are exactly what `encode_scalar_value_into` writes
//! for that type, so a field decodes through the same `decode_fixed_scalar`
//! and `decode_varlen_scalar` every other stored value uses. Nothing here
//! invents a second encoding for a scalar.

use zyron_catalog::schema::{NestedShape, NestedType};
use zyron_common::nested_value::{MapView, StructView, encode_map, encode_struct};
use zyron_common::{Result, TypeId, ZyronError};

use crate::batch::{decode_fixed_scalar, decode_varlen_scalar, encode_scalar_value_into};
use crate::column::ScalarValue;

/// The declared position of a field, which is how the binary form addresses
/// it. Names are matched without case, the way the binder resolves them.
pub fn field_ordinal(shape: &NestedShape, name: &str) -> Option<usize> {
    match shape {
        NestedShape::Struct(fields) => fields
            .iter()
            .position(|(field, _)| field.eq_ignore_ascii_case(name)),
        NestedShape::Map { .. } => None,
    }
}

/// The declared type reached by one lookup on a shape.
pub fn looked_up_type<'a>(shape: &'a NestedShape, name: &str) -> Option<&'a NestedType> {
    shape.lookup(name)
}

/// Bytes for one scalar, in the encoding every stored value of that type uses.
fn encode_leaf(type_id: TypeId, scalar: &ScalarValue) -> Vec<u8> {
    let value_size = type_id.fixed_size().unwrap_or(0);
    let mut buf = Vec::with_capacity(value_size.max(8));
    encode_scalar_value_into(&mut buf, type_id, scalar, value_size);
    buf
}

/// One scalar back out of a field's bytes.
fn decode_leaf(type_id: TypeId, bytes: &[u8]) -> ScalarValue {
    match type_id.fixed_size() {
        Some(size) if size > 0 => decode_fixed_scalar(type_id, bytes),
        _ => decode_varlen_scalar(type_id, bytes),
    }
}

/// The scalar a JSON value carries, coerced to `type_id`.
///
/// JSON has three scalar kinds and SQL has many, so the text of the value is
/// what gets coerced. That is the same route a literal takes, so a field
/// holds what the same value written to a plain column of that type would.
fn json_scalar(value: &serde_json::Value, type_id: TypeId, path: &str) -> Result<ScalarValue> {
    let text = match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => {
            return Err(ZyronError::CheckViolation(format!(
                "{path} is declared {type_id} and was given {}",
                json_kind(other)
            )));
        }
    };
    if matches!(type_id, TypeId::Text | TypeId::Varchar | TypeId::Char) {
        return Ok(ScalarValue::Utf8(text));
    }
    crate::compute::cast_scalar(&ScalarValue::Utf8(text.clone()), type_id).map_err(|_| {
        ZyronError::CheckViolation(format!("{path} is declared {type_id} and was given {text}"))
    })
}

/// What a JSON value is, for an error that says what arrived.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Encodes one JSON value against a declared type, returning None for null.
fn encode_typed(
    value: &serde_json::Value,
    declared: &NestedType,
    path: &str,
) -> Result<Option<Vec<u8>>> {
    if value.is_null() {
        return Ok(None);
    }
    if let Some(shape) = declared.shape.as_deref() {
        return Ok(Some(encode_shaped(value, shape, path)?));
    }
    if declared.type_id == TypeId::Array {
        let Some(elements) = value.as_array() else {
            return Err(ZyronError::CheckViolation(format!(
                "{path} is declared an array and was given {}",
                json_kind(value)
            )));
        };
        // An array with no declared element type holds text, which is the
        // rendering every scalar survives
        let fallback = NestedType::scalar(TypeId::Text);
        let element = declared.element.as_deref().unwrap_or(&fallback);
        let mut cells: Vec<Option<Vec<u8>>> = Vec::with_capacity(elements.len());
        for (i, item) in elements.iter().enumerate() {
            cells.push(encode_typed(item, element, &format!("{path}[{i}]"))?);
        }
        let borrowed: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
        return Ok(Some(zyron_common::array_value::encode(
            element.type_id,
            &borrowed,
        )));
    }
    if value.is_object() || value.is_array() {
        return Err(ZyronError::CheckViolation(format!(
            "{path} is declared {} and was given {}",
            declared.type_id,
            json_kind(value)
        )));
    }
    let scalar = json_scalar(value, declared.type_id, path)?;
    Ok(Some(encode_leaf(declared.type_id, &scalar)))
}

/// Encodes one JSON value against a declared STRUCT or MAP shape.
fn encode_shaped(value: &serde_json::Value, shape: &NestedShape, path: &str) -> Result<Vec<u8>> {
    match shape {
        NestedShape::Struct(fields) => {
            let Some(object) = value.as_object() else {
                return Err(ZyronError::CheckViolation(format!(
                    "{path} is declared a struct and was given {}",
                    json_kind(value)
                )));
            };
            // A field the declaration does not name could never be read back,
            // so storing it would discard data the writer believed it wrote
            for key in object.keys() {
                if !fields
                    .iter()
                    .any(|(name, _)| name.eq_ignore_ascii_case(key))
                {
                    return Err(ZyronError::CheckViolation(format!(
                        "{path} has no field \"{key}\", and a field the declaration does not name could never be read back"
                    )));
                }
            }
            let mut cells: Vec<Option<Vec<u8>>> = Vec::with_capacity(fields.len());
            for (name, declared) in fields {
                let found = object
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, v)| v);
                // A declared field the value omits is null, the same as an
                // absent field anywhere else
                match found {
                    Some(field) => {
                        cells.push(encode_typed(field, declared, &format!("{path}.{name}"))?)
                    }
                    None => cells.push(None),
                }
            }
            let borrowed: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
            Ok(encode_struct(&borrowed))
        }
        NestedShape::Map { key, value: val } => {
            let Some(object) = value.as_object() else {
                return Err(ZyronError::CheckViolation(format!(
                    "{path} is declared a map and was given {}",
                    json_kind(value)
                )));
            };
            let mut keys: Vec<Vec<u8>> = Vec::with_capacity(object.len());
            let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(object.len());
            for (entry_key, entry_value) in object {
                // Json names every key with text, so a map declaring another
                // key type has to be able to read its keys back as that type
                let scalar = crate::compute::cast_scalar(
                    &ScalarValue::Utf8(entry_key.clone()),
                    key.type_id,
                )
                .map_err(|_| {
                    ZyronError::CheckViolation(format!(
                        "{path} is declared with {} keys and was given the key \"{entry_key}\"",
                        key.type_id
                    ))
                })?;
                keys.push(encode_leaf(key.type_id, &scalar));
                values.push(encode_typed(
                    entry_value,
                    val,
                    &format!("{path}.{entry_key}"),
                )?);
            }
            let entries: Vec<(&[u8], Option<&[u8]>)> = keys
                .iter()
                .map(|k| k.as_slice())
                .zip(values.iter().map(|v| v.as_deref()))
                .collect();
            Ok(encode_map(&entries))
        }
    }
}

/// Encodes the JSON text a statement wrote into the column's binary form.
pub fn encode_json_text(text: &str, shape: &NestedShape, column: &str) -> Result<Vec<u8>> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| {
        ZyronError::CheckViolation(format!(
            "column \"{column}\" holds a nested value and was given text that is not json, {e}"
        ))
    })?;
    encode_shaped(&value, shape, column)
}

// ---------------------------------------------------------------------------
// Rendering back to JSON
// ---------------------------------------------------------------------------

/// Writes one scalar as JSON, quoting only what JSON quotes.
fn render_leaf(type_id: TypeId, bytes: &[u8], out: &mut String) {
    let scalar = decode_leaf(type_id, bytes);
    match &scalar {
        ScalarValue::Null => out.push_str("null"),
        ScalarValue::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        ScalarValue::Int8(_)
        | ScalarValue::Int16(_)
        | ScalarValue::Int32(_)
        | ScalarValue::Int64(_)
        | ScalarValue::Int128(_)
        | ScalarValue::UInt8(_)
        | ScalarValue::UInt16(_)
        | ScalarValue::UInt32(_)
        | ScalarValue::UInt64(_)
        | ScalarValue::Float32(_)
        | ScalarValue::Float64(_) => out.push_str(&scalar.to_string()),
        _ => {
            let text = scalar.to_string();
            out.push_str(&serde_json::Value::String(text).to_string());
        }
    }
}

/// Writes one encoded value as JSON against its declared type.
fn render_typed(bytes: Option<&[u8]>, declared: &NestedType, out: &mut String) {
    let Some(bytes) = bytes else {
        out.push_str("null");
        return;
    };
    if let Some(shape) = declared.shape.as_deref() {
        render_shaped(bytes, shape, out);
        return;
    }
    if declared.type_id == TypeId::Array {
        let fallback = NestedType::scalar(TypeId::Text);
        let element = declared.element.as_deref().unwrap_or(&fallback);
        let Some(view) = zyron_common::ArrayView::parse(bytes) else {
            out.push_str("null");
            return;
        };
        out.push('[');
        for i in 0..view.len() {
            if i > 0 {
                out.push(',');
            }
            render_typed(view.get(i).flatten(), element, out);
        }
        out.push(']');
        return;
    }
    render_leaf(declared.type_id, bytes, out);
}

/// Writes one encoded STRUCT or MAP as JSON against its declared shape.
fn render_shaped(bytes: &[u8], shape: &NestedShape, out: &mut String) {
    match shape {
        NestedShape::Struct(fields) => {
            let Some(view) = StructView::parse(bytes) else {
                out.push_str("null");
                return;
            };
            out.push('{');
            let mut first = true;
            for (i, (name, declared)) in fields.iter().enumerate() {
                // A field the value never held reads null, and a struct
                // renders every declared field so the shape is visible in
                // the answer
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&serde_json::Value::String(name.clone()).to_string());
                out.push(':');
                render_typed(view.field(i), declared, out);
            }
            out.push('}');
        }
        NestedShape::Map { key, value } => {
            let Some(view) = MapView::parse(bytes) else {
                out.push_str("null");
                return;
            };
            out.push('{');
            for i in 0..view.len() {
                if i > 0 {
                    out.push(',');
                }
                let key_text = match view.key(i) {
                    Some(k) => decode_leaf(key.type_id, k).to_string(),
                    None => String::new(),
                };
                out.push_str(&serde_json::Value::String(key_text).to_string());
                out.push(':');
                render_typed(view.value(i), value, out);
            }
            out.push('}');
        }
    }
}

/// Renders a stored nested value as the JSON text a client reads.
pub fn render_json_text(bytes: &[u8], shape: &NestedShape) -> String {
    let mut out = String::new();
    render_shaped(bytes, shape, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Field access
// ---------------------------------------------------------------------------

/// Reads one declared field out of an encoded struct, by position.
///
/// The result is the field's own scalar for a leaf, and the nested value's
/// bytes for a struct, map or array, which is what lets a deeper path chain
/// straight onto it.
pub fn struct_field(bytes: &[u8], ordinal: usize, declared: &NestedType) -> ScalarValue {
    let Some(view) = StructView::parse(bytes) else {
        return ScalarValue::Null;
    };
    match view.field(ordinal) {
        None => ScalarValue::Null,
        Some(field) => nested_scalar(field, declared),
    }
}

/// Reads one key out of an encoded map.
pub fn map_value(bytes: &[u8], key: &[u8], declared: &NestedType) -> ScalarValue {
    let Some(view) = MapView::parse(bytes) else {
        return ScalarValue::Null;
    };
    match view.lookup(key) {
        Some(Some(found)) => nested_scalar(found, declared),
        // An absent key and a present null key both read null, which is what
        // an absent field means everywhere else
        _ => ScalarValue::Null,
    }
}

/// The scalar a field's bytes carry: the nested bytes themselves when the
/// field is a struct, map or array, the decoded value otherwise.
fn nested_scalar(bytes: &[u8], declared: &NestedType) -> ScalarValue {
    if declared.shape.is_some() || declared.type_id == TypeId::Array {
        return ScalarValue::Binary(bytes.to_vec());
    }
    decode_leaf(declared.type_id, bytes)
}

/// Reads one field of an encoded struct by position, decoding it as
/// `result_type`. A field that is itself nested comes back as its own bytes,
/// which is what lets a deeper step chain straight onto it.
pub fn field_scalar(bytes: &[u8], ordinal: usize, result_type: TypeId) -> ScalarValue {
    let Some(view) = StructView::parse(bytes) else {
        return ScalarValue::Null;
    };
    match view.field(ordinal) {
        None => ScalarValue::Null,
        Some(field) => scalar_of(field, result_type),
    }
}

/// Reads one entry of an encoded map by key, decoding it as `result_type`.
pub fn key_scalar(bytes: &[u8], key: &[u8], result_type: TypeId) -> ScalarValue {
    let Some(view) = MapView::parse(bytes) else {
        return ScalarValue::Null;
    };
    match view.lookup(key) {
        Some(Some(found)) => scalar_of(found, result_type),
        // An absent key and a present null both read null, which is what an
        // absent field means everywhere else
        _ => ScalarValue::Null,
    }
}

/// The scalar a field's bytes carry for a result type: the bytes themselves
/// when the field is itself nested, the decoded value otherwise.
fn scalar_of(bytes: &[u8], result_type: TypeId) -> ScalarValue {
    if matches!(result_type, TypeId::Struct | TypeId::Map | TypeId::Array) {
        return ScalarValue::Binary(bytes.to_vec());
    }
    decode_leaf(result_type, bytes)
}

/// Whether bytes already carry the binary form rather than the json a
/// statement wrote. A shaped value is an object at its top level, so its json
/// always leads with a brace and never with either tag.
pub fn is_encoded(bytes: &[u8]) -> bool {
    match bytes.first() {
        Some(&zyron_common::nested_value::STRUCT_TAG) => StructView::parse(bytes).is_some(),
        Some(&zyron_common::nested_value::MAP_TAG) => MapView::parse(bytes).is_some(),
        _ => false,
    }
}

/// Encodes a map key for lookup, in the same encoding the value stores.
pub fn encode_map_key(key_text: &str, key_type: TypeId) -> Option<Vec<u8>> {
    let scalar = if matches!(key_type, TypeId::Text | TypeId::Varchar | TypeId::Char) {
        ScalarValue::Utf8(key_text.to_string())
    } else {
        crate::compute::cast_scalar(&ScalarValue::Utf8(key_text.to_string()), key_type).ok()?
    };
    Some(encode_leaf(key_type, &scalar))
}
