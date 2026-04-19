//! JSON Schema validation (draft 2020-12 subset).
//!
//! Supports: type, required, properties, additionalProperties, items,
//! minItems, maxItems, uniqueItems, minimum, maximum, exclusiveMinimum,
//! exclusiveMaximum, multipleOf, minLength, maxLength, pattern,
//! enum, const, oneOf, anyOf, allOf, not, and local $ref.

use crate::diff::JsonValue;
use zyron_common::Result;

/// Validates JSON against a schema. Returns true if valid.
pub fn json_schema_validate(json: &str, schema: &str) -> Result<bool> {
    let errors = json_schema_errors(json, schema)?;
    Ok(errors.is_empty())
}

/// Returns a list of validation error messages. Empty if valid.
pub fn json_schema_errors(json: &str, schema: &str) -> Result<Vec<String>> {
    let value = JsonValue::parse(json)?;
    let schema_val = JsonValue::parse(schema)?;
    let mut errors = Vec::new();
    validate_against_schema(&value, &schema_val, "$", &schema_val, &mut errors);
    Ok(errors)
}

fn validate_against_schema(
    value: &JsonValue,
    schema: &JsonValue,
    path: &str,
    root_schema: &JsonValue,
    errors: &mut Vec<String>,
) {
    let schema_obj = match schema {
        JsonValue::Object(items) => items,
        // "true" schema accepts anything, "false" rejects everything
        JsonValue::Bool(true) => return,
        JsonValue::Bool(false) => {
            errors.push(format!("{}: schema is false, nothing matches", path));
            return;
        }
        _ => {
            errors.push(format!("{}: schema must be an object or boolean", path));
            return;
        }
    };

    // Handle $ref (local only, #/path/to/def)
    if let Some((_, JsonValue::String(ref_path))) = schema_obj.iter().find(|(k, _)| k == "$ref") {
        if let Some(resolved) = resolve_ref(ref_path, root_schema) {
            validate_against_schema(value, &resolved, path, root_schema, errors);
            return;
        } else {
            errors.push(format!("{}: unresolved $ref: {}", path, ref_path));
            return;
        }
    }

    // const: exact match
    if let Some((_, const_val)) = schema_obj.iter().find(|(k, _)| k == "const") {
        if value != const_val {
            errors.push(format!("{}: expected const value", path));
        }
    }

    // enum: value must be in list
    if let Some((_, JsonValue::Array(enum_vals))) = schema_obj.iter().find(|(k, _)| k == "enum") {
        if !enum_vals.iter().any(|ev| ev == value) {
            errors.push(format!("{}: value not in enum", path));
        }
    }

    // type
    if let Some((_, type_val)) = schema_obj.iter().find(|(k, _)| k == "type") {
        let type_ok = match type_val {
            JsonValue::String(t) => matches_type(value, t),
            JsonValue::Array(types) => types.iter().any(|t| {
                if let JsonValue::String(s) = t {
                    matches_type(value, s)
                } else {
                    false
                }
            }),
            _ => false,
        };
        if !type_ok {
            errors.push(format!("{}: type mismatch", path));
        }
    }

    // Numeric constraints
    if let JsonValue::Number(n) = value {
        if let Some((_, JsonValue::Number(min))) = schema_obj.iter().find(|(k, _)| k == "minimum") {
            if n < min {
                errors.push(format!("{}: below minimum {}", path, min));
            }
        }
        if let Some((_, JsonValue::Number(max))) = schema_obj.iter().find(|(k, _)| k == "maximum") {
            if n > max {
                errors.push(format!("{}: above maximum {}", path, max));
            }
        }
        if let Some((_, JsonValue::Number(emin))) =
            schema_obj.iter().find(|(k, _)| k == "exclusiveMinimum")
        {
            if n <= emin {
                errors.push(format!("{}: not exclusively above {}", path, emin));
            }
        }
        if let Some((_, JsonValue::Number(emax))) =
            schema_obj.iter().find(|(k, _)| k == "exclusiveMaximum")
        {
            if n >= emax {
                errors.push(format!("{}: not exclusively below {}", path, emax));
            }
        }
        if let Some((_, JsonValue::Number(m))) = schema_obj.iter().find(|(k, _)| k == "multipleOf")
        {
            if *m != 0.0 && (n / m).fract().abs() > 1e-10 {
                errors.push(format!("{}: not a multiple of {}", path, m));
            }
        }
    }

    // String constraints
    if let JsonValue::String(s) = value {
        if let Some((_, JsonValue::Number(min))) = schema_obj.iter().find(|(k, _)| k == "minLength")
        {
            if (s.chars().count() as f64) < *min {
                errors.push(format!("{}: below minLength {}", path, min));
            }
        }
        if let Some((_, JsonValue::Number(max))) = schema_obj.iter().find(|(k, _)| k == "maxLength")
        {
            if (s.chars().count() as f64) > *max {
                errors.push(format!("{}: above maxLength {}", path, max));
            }
        }
        if let Some((_, JsonValue::String(pattern))) =
            schema_obj.iter().find(|(k, _)| k == "pattern")
        {
            if let Ok(matched) = crate::regex_type::regex_match(s, pattern) {
                if !matched {
                    errors.push(format!("{}: pattern mismatch", path));
                }
            }
        }
    }

    // Array constraints
    if let JsonValue::Array(items) = value {
        if let Some((_, JsonValue::Number(min))) = schema_obj.iter().find(|(k, _)| k == "minItems")
        {
            if (items.len() as f64) < *min {
                errors.push(format!("{}: fewer than minItems {}", path, min));
            }
        }
        if let Some((_, JsonValue::Number(max))) = schema_obj.iter().find(|(k, _)| k == "maxItems")
        {
            if (items.len() as f64) > *max {
                errors.push(format!("{}: more than maxItems {}", path, max));
            }
        }
        if let Some((_, JsonValue::Bool(true))) =
            schema_obj.iter().find(|(k, _)| k == "uniqueItems")
        {
            for (i, item_a) in items.iter().enumerate() {
                for item_b in &items[..i] {
                    if item_a == item_b {
                        errors.push(format!("{}: uniqueItems violated", path));
                        break;
                    }
                }
            }
        }
        if let Some((_, items_schema)) = schema_obj.iter().find(|(k, _)| k == "items") {
            for (i, item) in items.iter().enumerate() {
                let item_path = format!("{}[{}]", path, i);
                validate_against_schema(item, items_schema, &item_path, root_schema, errors);
            }
        }
    }

    // Object constraints
    if let JsonValue::Object(obj_items) = value {
        if let Some((_, JsonValue::Array(required))) =
            schema_obj.iter().find(|(k, _)| k == "required")
        {
            for req in required {
                if let JsonValue::String(req_name) = req {
                    if !obj_items.iter().any(|(k, _)| k == req_name) {
                        errors.push(format!("{}: missing required field '{}'", path, req_name));
                    }
                }
            }
        }

        if let Some((_, JsonValue::Object(props))) =
            schema_obj.iter().find(|(k, _)| k == "properties")
        {
            for (prop_name, prop_schema) in props {
                if let Some((_, prop_val)) = obj_items.iter().find(|(k, _)| k == prop_name) {
                    let prop_path = format!("{}.{}", path, prop_name);
                    validate_against_schema(prop_val, prop_schema, &prop_path, root_schema, errors);
                }
            }
        }

        // additionalProperties
        if let Some((_, add_props_schema)) =
            schema_obj.iter().find(|(k, _)| k == "additionalProperties")
        {
            let defined_props: Vec<String> = match schema_obj
                .iter()
                .find(|(k, _)| k == "properties")
                .map(|(_, v)| v)
            {
                Some(JsonValue::Object(p)) => p.iter().map(|(k, _)| k.clone()).collect(),
                _ => Vec::new(),
            };

            for (k, v) in obj_items {
                if !defined_props.contains(k) {
                    let add_path = format!("{}.{}", path, k);
                    match add_props_schema {
                        JsonValue::Bool(false) => {
                            errors.push(format!("{}: additional property not allowed", add_path));
                        }
                        JsonValue::Bool(true) => {} // anything goes
                        _ => {
                            validate_against_schema(
                                v,
                                add_props_schema,
                                &add_path,
                                root_schema,
                                errors,
                            );
                        }
                    }
                }
            }
        }
    }

    // allOf: all must match
    if let Some((_, JsonValue::Array(schemas))) = schema_obj.iter().find(|(k, _)| k == "allOf") {
        for s in schemas {
            validate_against_schema(value, s, path, root_schema, errors);
        }
    }

    // anyOf: at least one must match
    if let Some((_, JsonValue::Array(schemas))) = schema_obj.iter().find(|(k, _)| k == "anyOf") {
        let matches = schemas.iter().any(|s| {
            let mut temp_errors = Vec::new();
            validate_against_schema(value, s, path, root_schema, &mut temp_errors);
            temp_errors.is_empty()
        });
        if !matches {
            errors.push(format!("{}: matched no anyOf schema", path));
        }
    }

    // oneOf: exactly one must match
    if let Some((_, JsonValue::Array(schemas))) = schema_obj.iter().find(|(k, _)| k == "oneOf") {
        let match_count = schemas
            .iter()
            .filter(|s| {
                let mut temp_errors = Vec::new();
                validate_against_schema(value, s, path, root_schema, &mut temp_errors);
                temp_errors.is_empty()
            })
            .count();
        if match_count != 1 {
            errors.push(format!(
                "{}: matched {} oneOf schemas (expected exactly 1)",
                path, match_count
            ));
        }
    }

    // not: must not match
    if let Some((_, not_schema)) = schema_obj.iter().find(|(k, _)| k == "not") {
        let mut temp_errors = Vec::new();
        validate_against_schema(value, not_schema, path, root_schema, &mut temp_errors);
        if temp_errors.is_empty() {
            errors.push(format!("{}: matched not-schema", path));
        }
    }
}

fn matches_type(value: &JsonValue, type_name: &str) -> bool {
    match type_name {
        "null" => matches!(value, JsonValue::Null),
        "boolean" => matches!(value, JsonValue::Bool(_)),
        "integer" => {
            if let JsonValue::Number(n) = value {
                n.fract() == 0.0 && n.is_finite()
            } else {
                false
            }
        }
        "number" => matches!(value, JsonValue::Number(_)),
        "string" => matches!(value, JsonValue::String(_)),
        "array" => matches!(value, JsonValue::Array(_)),
        "object" => matches!(value, JsonValue::Object(_)),
        _ => false,
    }
}

fn resolve_ref(ref_path: &str, root: &JsonValue) -> Option<JsonValue> {
    let path = ref_path.strip_prefix("#/")?;
    let mut current = root;
    for seg in path.split('/') {
        let unescaped = seg.replace("~1", "/").replace("~0", "~");
        match current {
            JsonValue::Object(items) => {
                current = items
                    .iter()
                    .find(|(k, _)| k == &unescaped)
                    .map(|(_, v)| v)?;
            }
            JsonValue::Array(items) => {
                let idx: usize = unescaped.parse().ok()?;
                current = items.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(current.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_type_string() {
        assert!(json_schema_validate("\"hello\"", r#"{"type":"string"}"#).unwrap());
        assert!(!json_schema_validate("42", r#"{"type":"string"}"#).unwrap());
    }

    #[test]
    fn test_validate_type_number() {
        assert!(json_schema_validate("42", r#"{"type":"number"}"#).unwrap());
        assert!(json_schema_validate("3.14", r#"{"type":"number"}"#).unwrap());
        assert!(!json_schema_validate("\"hello\"", r#"{"type":"number"}"#).unwrap());
    }

    #[test]
    fn test_validate_type_integer() {
        assert!(json_schema_validate("42", r#"{"type":"integer"}"#).unwrap());
        assert!(!json_schema_validate("3.14", r#"{"type":"integer"}"#).unwrap());
    }

    #[test]
    fn test_validate_minimum() {
        assert!(json_schema_validate("10", r#"{"minimum":5}"#).unwrap());
        assert!(!json_schema_validate("3", r#"{"minimum":5}"#).unwrap());
    }

    #[test]
    fn test_validate_maximum() {
        assert!(json_schema_validate("3", r#"{"maximum":5}"#).unwrap());
        assert!(!json_schema_validate("10", r#"{"maximum":5}"#).unwrap());
    }

    #[test]
    fn test_validate_exclusive_minimum() {
        assert!(json_schema_validate("10", r#"{"exclusiveMinimum":5}"#).unwrap());
        assert!(!json_schema_validate("5", r#"{"exclusiveMinimum":5}"#).unwrap());
    }

    #[test]
    fn test_validate_multiple_of() {
        assert!(json_schema_validate("10", r#"{"multipleOf":5}"#).unwrap());
        assert!(!json_schema_validate("7", r#"{"multipleOf":5}"#).unwrap());
    }

    #[test]
    fn test_validate_min_length() {
        assert!(json_schema_validate("\"hello\"", r#"{"minLength":3}"#).unwrap());
        assert!(!json_schema_validate("\"hi\"", r#"{"minLength":3}"#).unwrap());
    }

    #[test]
    fn test_validate_max_length() {
        assert!(json_schema_validate("\"hi\"", r#"{"maxLength":5}"#).unwrap());
        assert!(!json_schema_validate("\"hello world\"", r#"{"maxLength":5}"#).unwrap());
    }

    #[test]
    fn test_validate_enum() {
        let schema = r#"{"enum":["red","green","blue"]}"#;
        assert!(json_schema_validate("\"red\"", schema).unwrap());
        assert!(!json_schema_validate("\"yellow\"", schema).unwrap());
    }

    #[test]
    fn test_validate_const() {
        let schema = r#"{"const":42}"#;
        assert!(json_schema_validate("42", schema).unwrap());
        assert!(!json_schema_validate("43", schema).unwrap());
    }

    #[test]
    fn test_validate_required() {
        let schema = r#"{"type":"object","required":["name","age"]}"#;
        assert!(json_schema_validate(r#"{"name":"Alice","age":30}"#, schema).unwrap());
        assert!(!json_schema_validate(r#"{"name":"Alice"}"#, schema).unwrap());
    }

    #[test]
    fn test_validate_properties() {
        let schema = r#"{
            "type":"object",
            "properties":{
                "name":{"type":"string"},
                "age":{"type":"integer","minimum":0}
            }
        }"#;
        assert!(json_schema_validate(r#"{"name":"Alice","age":30}"#, schema).unwrap());
        assert!(!json_schema_validate(r#"{"name":"Alice","age":-1}"#, schema).unwrap());
        assert!(!json_schema_validate(r#"{"name":42,"age":30}"#, schema).unwrap());
    }

    #[test]
    fn test_validate_additional_properties_false() {
        let schema = r#"{
            "type":"object",
            "properties":{"name":{"type":"string"}},
            "additionalProperties":false
        }"#;
        assert!(json_schema_validate(r#"{"name":"Alice"}"#, schema).unwrap());
        assert!(!json_schema_validate(r#"{"name":"Alice","extra":1}"#, schema).unwrap());
    }

    #[test]
    fn test_validate_array_items() {
        let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
        assert!(json_schema_validate("[1,2,3]", schema).unwrap());
        assert!(!json_schema_validate("[1,\"two\",3]", schema).unwrap());
    }

    #[test]
    fn test_validate_min_items() {
        let schema = r#"{"type":"array","minItems":2}"#;
        assert!(json_schema_validate("[1,2]", schema).unwrap());
        assert!(!json_schema_validate("[1]", schema).unwrap());
    }

    #[test]
    fn test_validate_unique_items() {
        let schema = r#"{"type":"array","uniqueItems":true}"#;
        assert!(json_schema_validate("[1,2,3]", schema).unwrap());
        assert!(!json_schema_validate("[1,2,1]", schema).unwrap());
    }

    #[test]
    fn test_validate_any_of() {
        let schema = r#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#;
        assert!(json_schema_validate("\"hello\"", schema).unwrap());
        assert!(json_schema_validate("42", schema).unwrap());
        assert!(!json_schema_validate("[1,2]", schema).unwrap());
    }

    #[test]
    fn test_validate_one_of() {
        let schema = r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#;
        assert!(json_schema_validate("\"hello\"", schema).unwrap());
        assert!(json_schema_validate("42", schema).unwrap());
    }

    #[test]
    fn test_validate_not() {
        let schema = r#"{"not":{"type":"string"}}"#;
        assert!(json_schema_validate("42", schema).unwrap());
        assert!(!json_schema_validate("\"hello\"", schema).unwrap());
    }

    #[test]
    fn test_validate_errors_returned() {
        let schema = r#"{"type":"object","required":["name"]}"#;
        let errors = json_schema_errors("{}", schema).unwrap();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_validate_pattern() {
        let schema = r#"{"type":"string","pattern":"^[A-Z]"}"#;
        assert!(json_schema_validate("\"Hello\"", schema).unwrap());
        assert!(!json_schema_validate("\"hello\"", schema).unwrap());
    }

    #[test]
    fn test_validate_nested() {
        let schema = r#"{
            "type":"object",
            "properties":{
                "user":{
                    "type":"object",
                    "properties":{
                        "email":{"type":"string"}
                    },
                    "required":["email"]
                }
            },
            "required":["user"]
        }"#;
        assert!(json_schema_validate(r#"{"user":{"email":"test@example.com"}}"#, schema).unwrap());
        assert!(!json_schema_validate(r#"{"user":{}}"#, schema).unwrap());
    }
}
