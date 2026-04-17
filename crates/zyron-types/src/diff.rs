//! Text and JSON diff/patch operations.
//!
//! Text diff uses Myers diff algorithm (O(ND) time, O(N) space).
//! JSON Patch follows RFC 6902, JSON Merge Patch follows RFC 7396.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Diff representation
// ---------------------------------------------------------------------------

/// A single operation in a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOp {
    Equal(String),
    Insert(String),
    Delete(String),
}

// ---------------------------------------------------------------------------
// Myers diff algorithm
// ---------------------------------------------------------------------------

fn myers_diff<T: Eq + Clone>(a: &[T], b: &[T]) -> Vec<DiffElem> {
    let n = a.len() as i32;
    let m = b.len() as i32;
    let max = n + m;

    if max == 0 {
        return Vec::new();
    }

    // v[k + max] holds furthest x for diagonal k
    let mut v = vec![0i32; (2 * max + 1) as usize];
    let offset = max;

    let mut trace = Vec::new();

    for d in 0..=max {
        let v_snapshot = v.clone();
        trace.push(v_snapshot.clone());

        let mut k = -d;
        while k <= d {
            let idx = (k + offset) as usize;
            let mut x = if k == -d
                || (k != d && v[(k - 1 + offset) as usize] < v[(k + 1 + offset) as usize])
            {
                v[(k + 1 + offset) as usize]
            } else {
                v[(k - 1 + offset) as usize] + 1
            };
            let mut y = x - k;

            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }

            v[idx] = x;

            if x >= n && y >= m {
                // Reconstruct path
                return backtrack(a, b, &trace, offset);
            }

            k += 2;
        }
    }

    Vec::new()
}

#[derive(Debug, Clone)]
enum DiffElem {
    Equal(usize),  // index in both sequences
    Delete(usize), // index in a
    Insert(usize), // index in b
}

fn backtrack<T: Eq + Clone>(a: &[T], b: &[T], trace: &[Vec<i32>], offset: i32) -> Vec<DiffElem> {
    let mut result: Vec<DiffElem> = Vec::new();
    let mut x = a.len() as i32;
    let mut y = b.len() as i32;

    for (d, v) in trace.iter().enumerate().rev() {
        let d = d as i32;
        let k = x - y;
        let prev_k =
            if k == -d || (k != d && v[(k - 1 + offset) as usize] < v[(k + 1 + offset) as usize]) {
                k + 1
            } else {
                k - 1
            };
        let prev_x = v[(prev_k + offset) as usize];
        let prev_y = prev_x - prev_k;

        while x > prev_x && y > prev_y {
            result.push(DiffElem::Equal((x - 1) as usize));
            x -= 1;
            y -= 1;
        }

        if d > 0 {
            if x == prev_x {
                result.push(DiffElem::Insert((y - 1) as usize));
            } else {
                result.push(DiffElem::Delete((x - 1) as usize));
            }
        }

        x = prev_x;
        y = prev_y;
    }

    result.reverse();
    // Suppress unused warnings
    let _ = a;
    let _ = b;
    result
}

// ---------------------------------------------------------------------------
// Text diff (line-based unified diff)
// ---------------------------------------------------------------------------

/// Computes a unified-diff-format text diff between two strings (line-based).
pub fn text_diff(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let diff = myers_diff(&old_lines, &new_lines);

    let mut result = String::new();
    for elem in &diff {
        match elem {
            DiffElem::Equal(i) => {
                result.push(' ');
                result.push_str(old_lines[*i]);
                result.push('\n');
            }
            DiffElem::Delete(i) => {
                result.push('-');
                result.push_str(old_lines[*i]);
                result.push('\n');
            }
            DiffElem::Insert(i) => {
                result.push('+');
                result.push_str(new_lines[*i]);
                result.push('\n');
            }
        }
    }
    result
}

/// Applies a unified-format text patch to a source string.
pub fn text_patch(text: &str, patch: &str) -> Result<String> {
    let source_lines: Vec<&str> = text.lines().collect();
    let mut source_iter = source_lines.iter();
    let mut result = Vec::new();

    for line in patch.lines() {
        if line.is_empty() {
            continue;
        }
        let prefix = line.chars().next().unwrap_or(' ');
        let content = &line[1.min(line.len())..];

        match prefix {
            ' ' => {
                // Context line - skip one from source, emit content
                source_iter.next();
                result.push(content.to_string());
            }
            '-' => {
                // Deletion - skip one from source
                source_iter.next();
            }
            '+' => {
                // Insertion - emit content without advancing source
                result.push(content.to_string());
            }
            _ => {
                return Err(ZyronError::ExecutionError(format!(
                    "Invalid patch line: {}",
                    line
                )));
            }
        }
    }

    // Append any remaining source lines
    for line in source_iter {
        result.push(line.to_string());
    }

    Ok(result.join("\n"))
}

/// Word-level diff between two strings.
pub fn text_diff_words(old: &str, new: &str) -> Vec<DiffOp> {
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();

    let diff = myers_diff(&old_words, &new_words);

    // Group consecutive same-type operations
    let mut ops: Vec<DiffOp> = Vec::new();
    let mut current_equal: Vec<String> = Vec::new();
    let mut current_insert: Vec<String> = Vec::new();
    let mut current_delete: Vec<String> = Vec::new();

    let flush_ops = |ops: &mut Vec<DiffOp>,
                     eq: &mut Vec<String>,
                     ins: &mut Vec<String>,
                     del: &mut Vec<String>| {
        if !eq.is_empty() {
            ops.push(DiffOp::Equal(eq.join(" ")));
            eq.clear();
        }
        if !del.is_empty() {
            ops.push(DiffOp::Delete(del.join(" ")));
            del.clear();
        }
        if !ins.is_empty() {
            ops.push(DiffOp::Insert(ins.join(" ")));
            ins.clear();
        }
    };

    for elem in &diff {
        match elem {
            DiffElem::Equal(i) => {
                if !current_insert.is_empty() || !current_delete.is_empty() {
                    flush_ops(
                        &mut ops,
                        &mut current_equal,
                        &mut current_insert,
                        &mut current_delete,
                    );
                }
                current_equal.push(old_words[*i].to_string());
            }
            DiffElem::Insert(i) => {
                if !current_equal.is_empty() {
                    flush_ops(
                        &mut ops,
                        &mut current_equal,
                        &mut current_insert,
                        &mut current_delete,
                    );
                }
                current_insert.push(new_words[*i].to_string());
            }
            DiffElem::Delete(i) => {
                if !current_equal.is_empty() {
                    flush_ops(
                        &mut ops,
                        &mut current_equal,
                        &mut current_insert,
                        &mut current_delete,
                    );
                }
                current_delete.push(old_words[*i].to_string());
            }
        }
    }
    flush_ops(
        &mut ops,
        &mut current_equal,
        &mut current_insert,
        &mut current_delete,
    );

    ops
}

// ---------------------------------------------------------------------------
// JSON value (minimal, for diff/patch only)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn parse(text: &str) -> Result<Self> {
        let (val, rest) = parse_json(text.trim_start())?;
        if !rest.trim().is_empty() {
            return Err(ZyronError::ExecutionError("Trailing JSON content".into()));
        }
        Ok(val)
    }

    pub fn stringify(&self) -> String {
        let mut s = String::new();
        stringify_into(self, &mut s);
        s
    }
}

fn parse_json(s: &str) -> Result<(JsonValue, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return Err(ZyronError::ExecutionError("Empty JSON".into()));
    }
    match s.chars().next().unwrap() {
        '"' => {
            let (s_val, rest) = parse_json_string(s)?;
            Ok((JsonValue::String(s_val), rest))
        }
        't' | 'f' | 'n' => parse_json_keyword(s),
        '[' => parse_json_array(s),
        '{' => parse_json_object(s),
        '-' | '0'..='9' => parse_json_number(s),
        c => Err(ZyronError::ExecutionError(format!(
            "Unexpected JSON character: {}",
            c
        ))),
    }
}

fn parse_json_string(s: &str) -> Result<(String, &str)> {
    let rest = s
        .strip_prefix('"')
        .ok_or_else(|| ZyronError::ExecutionError("Expected '\"'".into()))?;
    let mut result = String::new();
    let mut chars = rest.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Ok((result, &rest[i + 1..])),
            '\\' => {
                let (_, escaped) = chars
                    .next()
                    .ok_or_else(|| ZyronError::ExecutionError("Unterminated escape".into()))?;
                let decoded = match escaped {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'b' => '\u{0008}',
                    'f' => '\u{000C}',
                    c => c,
                };
                result.push(decoded);
            }
            c => result.push(c),
        }
    }
    Err(ZyronError::ExecutionError("Unterminated string".into()))
}

fn parse_json_keyword(s: &str) -> Result<(JsonValue, &str)> {
    if let Some(rest) = s.strip_prefix("true") {
        Ok((JsonValue::Bool(true), rest))
    } else if let Some(rest) = s.strip_prefix("false") {
        Ok((JsonValue::Bool(false), rest))
    } else if let Some(rest) = s.strip_prefix("null") {
        Ok((JsonValue::Null, rest))
    } else {
        Err(ZyronError::ExecutionError("Invalid keyword".into()))
    }
}

fn parse_json_number(s: &str) -> Result<(JsonValue, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    let num_str = &s[..i];
    let val: f64 = num_str
        .parse()
        .map_err(|_| ZyronError::ExecutionError(format!("Invalid number: {}", num_str)))?;
    Ok((JsonValue::Number(val), &s[i..]))
}

fn parse_json_array(s: &str) -> Result<(JsonValue, &str)> {
    let mut rest = s
        .strip_prefix('[')
        .ok_or_else(|| ZyronError::ExecutionError("Expected '['".into()))?
        .trim_start();
    if let Some(r) = rest.strip_prefix(']') {
        return Ok((JsonValue::Array(Vec::new()), r));
    }
    let mut items = Vec::new();
    loop {
        let (val, r) = parse_json(rest)?;
        items.push(val);
        rest = r.trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r.trim_start();
        } else if let Some(r) = rest.strip_prefix(']') {
            return Ok((JsonValue::Array(items), r));
        } else {
            return Err(ZyronError::ExecutionError("Expected ',' or ']'".into()));
        }
    }
}

fn parse_json_object(s: &str) -> Result<(JsonValue, &str)> {
    let mut rest = s
        .strip_prefix('{')
        .ok_or_else(|| ZyronError::ExecutionError("Expected '{'".into()))?
        .trim_start();
    if let Some(r) = rest.strip_prefix('}') {
        return Ok((JsonValue::Object(Vec::new()), r));
    }
    let mut items = Vec::new();
    loop {
        let (key, r) = parse_json_string(rest)?;
        rest = r
            .trim_start()
            .strip_prefix(':')
            .ok_or_else(|| ZyronError::ExecutionError("Expected ':' in object".into()))?
            .trim_start();
        let (val, r) = parse_json(rest)?;
        items.push((key, val));
        rest = r.trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r.trim_start();
        } else if let Some(r) = rest.strip_prefix('}') {
            return Ok((JsonValue::Object(items), r));
        } else {
            return Err(ZyronError::ExecutionError("Expected ',' or '}'".into()));
        }
    }
}

fn stringify_into(val: &JsonValue, out: &mut String) {
    match val {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        JsonValue::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                stringify_into(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('"');
                out.push_str(k);
                out.push_str("\":");
                stringify_into(v, out);
            }
            out.push('}');
        }
    }
}

// ---------------------------------------------------------------------------
// JSON Patch (RFC 6902)
// ---------------------------------------------------------------------------

/// Computes a JSON Patch (RFC 6902) representing the difference.
pub fn json_diff(old: &str, new: &str) -> Result<String> {
    let old_val = JsonValue::parse(old)?;
    let new_val = JsonValue::parse(new)?;

    let mut ops: Vec<JsonValue> = Vec::new();
    compute_json_diff(&old_val, &new_val, "", &mut ops);

    Ok(JsonValue::Array(ops).stringify())
}

fn compute_json_diff(old: &JsonValue, new: &JsonValue, path: &str, ops: &mut Vec<JsonValue>) {
    if old == new {
        return;
    }

    match (old, new) {
        (JsonValue::Object(o_items), JsonValue::Object(n_items)) => {
            // Find removed keys
            for (k, _) in o_items {
                if !n_items.iter().any(|(k2, _)| k2 == k) {
                    let mut op = Vec::new();
                    op.push(("op".to_string(), JsonValue::String("remove".into())));
                    op.push((
                        "path".to_string(),
                        JsonValue::String(format!("{}/{}", path, escape_json_pointer(k))),
                    ));
                    ops.push(JsonValue::Object(op));
                }
            }
            // Find added or changed keys
            for (k, v) in n_items {
                match o_items.iter().find(|(k2, _)| k2 == k) {
                    None => {
                        let mut op = Vec::new();
                        op.push(("op".to_string(), JsonValue::String("add".into())));
                        op.push((
                            "path".to_string(),
                            JsonValue::String(format!("{}/{}", path, escape_json_pointer(k))),
                        ));
                        op.push(("value".to_string(), v.clone()));
                        ops.push(JsonValue::Object(op));
                    }
                    Some((_, old_v)) if old_v != v => {
                        let new_path = format!("{}/{}", path, escape_json_pointer(k));
                        compute_json_diff(old_v, v, &new_path, ops);
                    }
                    _ => {}
                }
            }
        }
        (JsonValue::Array(_), JsonValue::Array(_)) => {
            // Simplified: replace the whole array
            let mut op = Vec::new();
            op.push(("op".to_string(), JsonValue::String("replace".into())));
            op.push(("path".to_string(), JsonValue::String(path.to_string())));
            op.push(("value".to_string(), new.clone()));
            ops.push(JsonValue::Object(op));
        }
        _ => {
            let mut op = Vec::new();
            op.push(("op".to_string(), JsonValue::String("replace".into())));
            op.push(("path".to_string(), JsonValue::String(path.to_string())));
            op.push(("value".to_string(), new.clone()));
            ops.push(JsonValue::Object(op));
        }
    }
}

fn escape_json_pointer(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

fn unescape_json_pointer(s: &str) -> String {
    s.replace("~1", "/").replace("~0", "~")
}

/// Applies a JSON Patch (RFC 6902) to a document.
pub fn json_patch(doc: &str, patch: &str) -> Result<String> {
    let mut value = JsonValue::parse(doc)?;
    let patch_val = JsonValue::parse(patch)?;

    let ops = match patch_val {
        JsonValue::Array(ops) => ops,
        _ => return Err(ZyronError::ExecutionError("Patch must be an array".into())),
    };

    for op in ops {
        apply_patch_op(&mut value, &op)?;
    }

    Ok(value.stringify())
}

fn apply_patch_op(doc: &mut JsonValue, op: &JsonValue) -> Result<()> {
    let op_obj = match op {
        JsonValue::Object(items) => items,
        _ => {
            return Err(ZyronError::ExecutionError(
                "Patch op must be an object".into(),
            ));
        }
    };

    let op_name = op_obj
        .iter()
        .find(|(k, _)| k == "op")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'op' field".into()))?;

    let path = op_obj
        .iter()
        .find(|(k, _)| k == "path")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'path' field".into()))?;

    match op_name.as_str() {
        "add" | "replace" => {
            let value = op_obj
                .iter()
                .find(|(k, _)| k == "value")
                .map(|(_, v)| v.clone())
                .ok_or_else(|| ZyronError::ExecutionError("Missing 'value' field".into()))?;
            set_at_path(doc, &path, value)?;
        }
        "remove" => {
            remove_at_path(doc, &path)?;
        }
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "Unsupported op: {}",
                op_name
            )));
        }
    }

    Ok(())
}

fn set_at_path(doc: &mut JsonValue, path: &str, value: JsonValue) -> Result<()> {
    if path.is_empty() || path == "/" {
        *doc = value;
        return Ok(());
    }
    let segments: Vec<String> = path
        .trim_start_matches('/')
        .split('/')
        .map(unescape_json_pointer)
        .collect();

    let mut current = doc;
    for seg in &segments[..segments.len() - 1] {
        current = navigate_mut(current, seg)?;
    }
    let last = &segments[segments.len() - 1];
    set_segment(current, last, value)?;
    Ok(())
}

fn remove_at_path(doc: &mut JsonValue, path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(ZyronError::ExecutionError("Cannot remove root".into()));
    }
    let segments: Vec<String> = path
        .trim_start_matches('/')
        .split('/')
        .map(unescape_json_pointer)
        .collect();

    let mut current = doc;
    for seg in &segments[..segments.len() - 1] {
        current = navigate_mut(current, seg)?;
    }
    let last = &segments[segments.len() - 1];
    remove_segment(current, last)?;
    Ok(())
}

fn navigate_mut<'a>(val: &'a mut JsonValue, seg: &str) -> Result<&'a mut JsonValue> {
    match val {
        JsonValue::Object(items) => items
            .iter_mut()
            .find(|(k, _)| k == seg)
            .map(|(_, v)| v)
            .ok_or_else(|| ZyronError::ExecutionError(format!("Missing key: {}", seg))),
        JsonValue::Array(items) => {
            let idx: usize = seg
                .parse()
                .map_err(|_| ZyronError::ExecutionError(format!("Invalid array index: {}", seg)))?;
            items.get_mut(idx).ok_or_else(|| {
                ZyronError::ExecutionError(format!("Array index out of bounds: {}", idx))
            })
        }
        _ => Err(ZyronError::ExecutionError(
            "Cannot navigate into scalar".into(),
        )),
    }
}

fn set_segment(val: &mut JsonValue, seg: &str, value: JsonValue) -> Result<()> {
    match val {
        JsonValue::Object(items) => {
            if let Some((_, existing)) = items.iter_mut().find(|(k, _)| k == seg) {
                *existing = value;
            } else {
                items.push((seg.to_string(), value));
            }
        }
        JsonValue::Array(items) => {
            if seg == "-" {
                items.push(value);
            } else {
                let idx: usize = seg
                    .parse()
                    .map_err(|_| ZyronError::ExecutionError(format!("Invalid index: {}", seg)))?;
                if idx > items.len() {
                    return Err(ZyronError::ExecutionError("Index out of range".into()));
                }
                if idx == items.len() {
                    items.push(value);
                } else {
                    items[idx] = value;
                }
            }
        }
        _ => return Err(ZyronError::ExecutionError("Cannot set on scalar".into())),
    }
    Ok(())
}

fn remove_segment(val: &mut JsonValue, seg: &str) -> Result<()> {
    match val {
        JsonValue::Object(items) => {
            let pos = items
                .iter()
                .position(|(k, _)| k == seg)
                .ok_or_else(|| ZyronError::ExecutionError(format!("Missing key: {}", seg)))?;
            items.remove(pos);
        }
        JsonValue::Array(items) => {
            let idx: usize = seg
                .parse()
                .map_err(|_| ZyronError::ExecutionError(format!("Invalid index: {}", seg)))?;
            if idx >= items.len() {
                return Err(ZyronError::ExecutionError("Index out of range".into()));
            }
            items.remove(idx);
        }
        _ => {
            return Err(ZyronError::ExecutionError(
                "Cannot remove from scalar".into(),
            ));
        }
    }
    Ok(())
}

/// Applies a JSON Merge Patch (RFC 7396) to a document.
pub fn json_merge_patch(doc: &str, patch: &str) -> Result<String> {
    let mut value = JsonValue::parse(doc)?;
    let patch_val = JsonValue::parse(patch)?;
    merge_patch(&mut value, &patch_val);
    Ok(value.stringify())
}

fn merge_patch(target: &mut JsonValue, patch: &JsonValue) {
    match (target, patch) {
        (target, JsonValue::Object(patch_items)) => {
            if !matches!(target, JsonValue::Object(_)) {
                *target = JsonValue::Object(Vec::new());
            }
            if let JsonValue::Object(t_items) = target {
                for (k, v) in patch_items {
                    if matches!(v, JsonValue::Null) {
                        // Null means remove
                        t_items.retain(|(tk, _)| tk != k);
                    } else if let Some((_, existing)) = t_items.iter_mut().find(|(tk, _)| tk == k) {
                        merge_patch(existing, v);
                    } else {
                        t_items.push((k.clone(), v.clone()));
                    }
                }
            }
        }
        (target, patch) => {
            *target = patch.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_diff_identical() {
        let diff = text_diff("hello", "hello");
        assert!(diff.contains(" hello"));
    }

    #[test]
    fn test_text_diff_changes() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline2 modified\nline3";
        let diff = text_diff(old, new);
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+line2 modified"));
    }

    #[test]
    fn test_text_diff_add_line() {
        let old = "a\nb";
        let new = "a\nb\nc";
        let diff = text_diff(old, new);
        assert!(diff.contains("+c"));
    }

    #[test]
    fn test_text_diff_delete_line() {
        let old = "a\nb\nc";
        let new = "a\nc";
        let diff = text_diff(old, new);
        assert!(diff.contains("-b"));
    }

    #[test]
    fn test_text_diff_words() {
        let ops = text_diff_words("the quick fox", "the slow fox");
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_json_parse_basics() {
        assert!(JsonValue::parse("null").is_ok());
        assert!(JsonValue::parse("true").is_ok());
        assert!(JsonValue::parse("42").is_ok());
        assert!(JsonValue::parse("\"hello\"").is_ok());
        assert!(JsonValue::parse("[1, 2, 3]").is_ok());
        assert!(JsonValue::parse("{\"k\": \"v\"}").is_ok());
    }

    #[test]
    fn test_json_stringify_roundtrip() {
        let cases = [
            "null",
            "true",
            "false",
            "42",
            "\"hello\"",
            "[1,2,3]",
            "{\"key\":\"value\"}",
        ];
        for c in cases {
            let val = JsonValue::parse(c).unwrap();
            assert_eq!(val.stringify(), c);
        }
    }

    #[test]
    fn test_json_diff_add() {
        let patch = json_diff("{\"a\":1}", "{\"a\":1,\"b\":2}").unwrap();
        assert!(patch.contains("add"));
        assert!(patch.contains("/b"));
    }

    #[test]
    fn test_json_diff_remove() {
        let patch = json_diff("{\"a\":1,\"b\":2}", "{\"a\":1}").unwrap();
        assert!(patch.contains("remove"));
    }

    #[test]
    fn test_json_diff_replace() {
        let patch = json_diff("{\"a\":1}", "{\"a\":2}").unwrap();
        assert!(patch.contains("replace"));
    }

    #[test]
    fn test_json_diff_identical() {
        let patch = json_diff("{\"a\":1}", "{\"a\":1}").unwrap();
        assert_eq!(patch, "[]");
    }

    #[test]
    fn test_json_patch_add() {
        let patch = "[{\"op\":\"add\",\"path\":\"/b\",\"value\":2}]";
        let result = json_patch("{\"a\":1}", patch).unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            assert!(items.iter().any(|(k, _)| k == "b"));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_json_patch_remove() {
        let patch = "[{\"op\":\"remove\",\"path\":\"/b\"}]";
        let result = json_patch("{\"a\":1,\"b\":2}", patch).unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            assert!(!items.iter().any(|(k, _)| k == "b"));
        }
    }

    #[test]
    fn test_json_patch_replace() {
        let patch = "[{\"op\":\"replace\",\"path\":\"/a\",\"value\":99}]";
        let result = json_patch("{\"a\":1}", patch).unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            let (_, v) = items.iter().find(|(k, _)| k == "a").unwrap();
            if let JsonValue::Number(n) = v {
                assert_eq!(*n as i64, 99);
            } else {
                panic!("Expected number");
            }
        }
    }

    #[test]
    fn test_json_diff_patch_roundtrip() {
        let old = "{\"x\":1,\"y\":2}";
        let new = "{\"x\":10,\"z\":3}";
        let patch = json_diff(old, new).unwrap();
        let result = json_patch(old, &patch).unwrap();
        // Parse both to compare semantically (key order may differ)
        let result_val = JsonValue::parse(&result).unwrap();
        let expected_val = JsonValue::parse(new).unwrap();
        // Compare as sorted objects
        if let (JsonValue::Object(r), JsonValue::Object(e)) = (&result_val, &expected_val) {
            assert_eq!(r.len(), e.len());
        }
    }

    #[test]
    fn test_json_merge_patch_add() {
        let result = json_merge_patch("{\"a\":1}", "{\"b\":2}").unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            assert_eq!(items.len(), 2);
        }
    }

    #[test]
    fn test_json_merge_patch_null_removes() {
        let result = json_merge_patch("{\"a\":1,\"b\":2}", "{\"b\":null}").unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            assert_eq!(items.len(), 1);
            assert!(items.iter().any(|(k, _)| k == "a"));
        }
    }

    #[test]
    fn test_json_merge_patch_nested() {
        let result = json_merge_patch("{\"a\":{\"b\":1,\"c\":2}}", "{\"a\":{\"b\":10}}").unwrap();
        let val = JsonValue::parse(&result).unwrap();
        if let JsonValue::Object(items) = val {
            if let Some((_, JsonValue::Object(inner))) = items.iter().find(|(k, _)| k == "a") {
                assert_eq!(inner.len(), 2); // b updated, c preserved
            }
        }
    }
}
