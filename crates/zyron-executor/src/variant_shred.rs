//! Process wide JSON path frequency tracking for variant columns, backing
//! `zyron_sys.storage.variant_shredding_stats` and the shredding promotion
//! decision
//!
//! Every variant write reports its JSON text here. The tracker counts how
//! often each dotted object path carries a scalar, per (table, column), so
//! paths that appear in most rows can be promoted to shredded columns. The
//! write path touches only scc maps and atomics, never a lock, and the
//! counters can be serialized and reloaded so a restart does not forget what
//! the workload looks like

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use zyron_common::{Result, ZyronError};

/// Most distinct paths tracked per column. A new path observed past the cap
/// is dropped rather than evicting an established one
const MAX_TRACKED_PATHS: usize = 256;

/// Deepest dotted path recorded, counted in segments
const MAX_PATH_DEPTH: usize = 4;

const SNAPSHOT_VERSION: u64 = 1;

/// One tracked path of one variant column, as the view and the promotion
/// scan read it
#[derive(Debug, Clone)]
pub struct PathStats {
    pub path: String,
    pub occurrences: u64,
    pub coverage_percent: f64,
    pub value_kind: &'static str,
    pub shredded: bool,
}

/// Which scalar kind one write carried at one path
#[derive(Clone, Copy)]
enum ScalarKind {
    String,
    Number,
    Bool,
}

struct PathCounter {
    occurrences: AtomicU64,
    string_count: AtomicU64,
    number_count: AtomicU64,
    bool_count: AtomicU64,
    shredded: AtomicBool,
}

impl PathCounter {
    fn new() -> Self {
        PathCounter {
            occurrences: AtomicU64::new(0),
            string_count: AtomicU64::new(0),
            number_count: AtomicU64::new(0),
            bool_count: AtomicU64::new(0),
            shredded: AtomicBool::new(false),
        }
    }

    fn bump(&self, kind: ScalarKind) {
        self.occurrences.fetch_add(1, Ordering::Relaxed);
        let slot = match kind {
            ScalarKind::String => &self.string_count,
            ScalarKind::Number => &self.number_count,
            ScalarKind::Bool => &self.bool_count,
        };
        slot.fetch_add(1, Ordering::Relaxed);
    }

    /// The kind seen most often at this path. A tie, including a path never
    /// observed with a scalar, reports string because text is the rendering
    /// every kind survives
    fn dominant_kind(&self) -> &'static str {
        let strings = self.string_count.load(Ordering::Relaxed);
        let numbers = self.number_count.load(Ordering::Relaxed);
        let bools = self.bool_count.load(Ordering::Relaxed);
        if numbers > strings && numbers > bools {
            "number"
        } else if bools > strings && bools > numbers {
            "bool"
        } else {
            "string"
        }
    }
}

struct ColumnTracker {
    total_writes: AtomicU64,
    malformed_writes: AtomicU64,
    paths: scc::HashMap<String, Arc<PathCounter>>,
    tracked_paths: AtomicUsize,
}

impl ColumnTracker {
    fn new() -> Self {
        ColumnTracker {
            total_writes: AtomicU64::new(0),
            malformed_writes: AtomicU64::new(0),
            paths: scc::HashMap::new(),
            tracked_paths: AtomicUsize::new(0),
        }
    }
}

static TRACKER: OnceLock<scc::HashMap<(u32, u16), Arc<ColumnTracker>>> = OnceLock::new();

fn tracker() -> &'static scc::HashMap<(u32, u16), Arc<ColumnTracker>> {
    TRACKER.get_or_init(scc::HashMap::new)
}

fn column_tracker(table_id: u32, column_id: u16) -> Arc<ColumnTracker> {
    let key = (table_id, column_id);
    if let Some(found) = tracker().read_sync(&key, |_, v| Arc::clone(v)) {
        return found;
    }
    match tracker().entry_sync(key) {
        scc::hash_map::Entry::Occupied(occupied) => Arc::clone(occupied.get()),
        scc::hash_map::Entry::Vacant(vacant) => {
            let created = Arc::new(ColumnTracker::new());
            vacant.insert_entry(Arc::clone(&created));
            created
        }
    }
}

fn column_if_tracked(table_id: u32, column_id: u16) -> Option<Arc<ColumnTracker>> {
    tracker().read_sync(&(table_id, column_id), |_, v| Arc::clone(v))
}

/// Records one write of a variant column. Scalar values at dotted object
/// paths up to four segments deep bump their path counter, any well formed
/// value bumps the column total, and text that does not parse bumps only the
/// malformed counter because the write path rejects it elsewhere
pub fn record_variant_write(table_id: u32, column_id: u16, json_text: &str) {
    let column = column_tracker(table_id, column_id);
    let parsed: serde_json::Value = match serde_json::from_str(json_text) {
        Ok(value) => value,
        Err(_) => {
            column.malformed_writes.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    column.total_writes.fetch_add(1, Ordering::Relaxed);
    if let serde_json::Value::Object(fields) = &parsed {
        record_object_paths(&column, fields, "", MAX_PATH_DEPTH);
    }
}

fn record_object_paths(
    column: &ColumnTracker,
    fields: &serde_json::Map<String, serde_json::Value>,
    prefix: &str,
    remaining_depth: usize,
) {
    for (key, value) in fields {
        // A key holding a dot would collide with the path separator, so the
        // path would be unaddressable and is not tracked
        if key.contains('.') {
            continue;
        }
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            serde_json::Value::String(_) => bump_path(column, path, ScalarKind::String),
            serde_json::Value::Number(_) => bump_path(column, path, ScalarKind::Number),
            serde_json::Value::Bool(_) => bump_path(column, path, ScalarKind::Bool),
            serde_json::Value::Object(nested) => {
                if remaining_depth > 1 {
                    record_object_paths(column, nested, &path, remaining_depth - 1);
                }
            }
            serde_json::Value::Array(_) | serde_json::Value::Null => {}
        }
    }
}

fn bump_path(column: &ColumnTracker, path: String, kind: ScalarKind) {
    if let Some(counter) = column.paths.read_sync(&path, |_, v| Arc::clone(v)) {
        counter.bump(kind);
        return;
    }
    match column.paths.entry_sync(path) {
        scc::hash_map::Entry::Occupied(occupied) => occupied.get().bump(kind),
        scc::hash_map::Entry::Vacant(vacant) => {
            // fetch_add reserves a slot below the cap, so concurrent first
            // observations can never insert a 257th path
            let reserved = column.tracked_paths.fetch_add(1, Ordering::Relaxed);
            if reserved >= MAX_TRACKED_PATHS {
                column.tracked_paths.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            let counter = Arc::new(PathCounter::new());
            counter.bump(kind);
            vacant.insert_entry(counter);
        }
    }
}

/// Every tracked path of one column, most frequent first
pub fn paths_for(table_id: u32, column_id: u16) -> Vec<PathStats> {
    let Some(column) = column_if_tracked(table_id, column_id) else {
        return Vec::new();
    };
    let total_writes = column.total_writes.load(Ordering::Relaxed);
    let mut out = Vec::new();
    column.paths.iter_sync(|path, counter| {
        let occurrences = counter.occurrences.load(Ordering::Relaxed);
        let coverage_percent = if total_writes == 0 {
            0.0
        } else {
            occurrences as f64 * 100.0 / total_writes as f64
        };
        out.push(PathStats {
            path: path.clone(),
            occurrences,
            coverage_percent,
            value_kind: counter.dominant_kind(),
            shredded: counter.shredded.load(Ordering::Relaxed),
        });
        true
    });
    out.sort_by(|a, b| {
        b.occurrences
            .cmp(&a.occurrences)
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

/// Paths not yet shredded that clear both thresholds, most frequent first
pub fn promotion_candidates(
    table_id: u32,
    column_id: u16,
    min_occurrences: u64,
    min_coverage_percent: f64,
) -> Vec<PathStats> {
    paths_for(table_id, column_id)
        .into_iter()
        .filter(|stats| {
            !stats.shredded
                && stats.occurrences >= min_occurrences
                && stats.coverage_percent >= min_coverage_percent
        })
        .collect()
}

/// Whether any path has been promoted in this process.
///
/// A shredded column exists only where a path was promoted, so a reader that
/// finds this false has nothing to look for and can skip the search entirely.
/// It never goes back to false once set, which costs a search that finds
/// nothing rather than a missed column
static ANY_SHREDDED: AtomicBool = AtomicBool::new(false);

/// Whether this process has promoted any variant path at all.
///
/// One relaxed load, so a deployment that shreds nothing pays nothing for
/// the machinery
#[inline]
pub fn any_shredded() -> bool {
    ANY_SHREDDED.load(Ordering::Relaxed)
}

/// Flags one path as promoted so it stops appearing as a candidate. A path
/// the tracker has not observed is created flagged, so the shredded set is
/// authoritative regardless of write order
pub fn mark_shredded(table_id: u32, column_id: u16, path: &str) {
    ANY_SHREDDED.store(true, Ordering::Relaxed);
    let column = column_tracker(table_id, column_id);
    let flagged = column.paths.read_sync(path, |_, counter| {
        counter.shredded.store(true, Ordering::Relaxed)
    });
    if flagged.is_some() {
        return;
    }
    match column.paths.entry_sync(path.to_string()) {
        scc::hash_map::Entry::Occupied(occupied) => {
            occupied.get().shredded.store(true, Ordering::Relaxed);
        }
        scc::hash_map::Entry::Vacant(vacant) => {
            column.tracked_paths.fetch_add(1, Ordering::Relaxed);
            let counter = PathCounter::new();
            counter.shredded.store(true, Ordering::Relaxed);
            vacant.insert_entry(Arc::new(counter));
        }
    }
}

/// The paths of one column flagged as shredded, sorted
pub fn shredded_paths(table_id: u32, column_id: u16) -> Vec<String> {
    let Some(column) = column_if_tracked(table_id, column_id) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    column.paths.iter_sync(|path, counter| {
        if counter.shredded.load(Ordering::Relaxed) {
            out.push(path.clone());
        }
        true
    });
    out.sort();
    out
}

/// Resolves a dotted path against JSON text. Returns None when the text does
/// not parse, the path is empty, or any step lands on a missing key or a non
/// object intermediate
pub fn extract_path(json_text: &str, dotted_path: &str) -> Option<serde_json::Value> {
    // Only the slice the path lands on is parsed, so reading one field of a
    // large document costs the walk to that field rather than a tree of the
    // whole document
    serde_json::from_str(extract_raw(json_text, dotted_path)?).ok()
}

/// The scalar at a dotted path rendered as SQL text output renders it,
/// strings without quotes, numbers as written, bools as true or false. None
/// for a missing value or one that is not a scalar
pub fn extract_scalar_text(json_text: &str, dotted_path: &str) -> Option<String> {
    let raw = extract_raw(json_text, dotted_path)?;
    match raw.as_bytes().first()? {
        // A string decodes its escapes, and one carrying none is handed
        // back without a pass over its bytes
        b'"' => {
            let inner = &raw[1..raw.len().checked_sub(1)?];
            if inner.as_bytes().contains(&b'\\') {
                serde_json::from_str::<String>(raw).ok()
            } else {
                Some(inner.to_string())
            }
        }
        b'{' | b'[' => None,
        _ => match raw {
            "true" | "false" => Some(raw.to_string()),
            "null" => None,
            // A number renders the way the JSON decoder renders it, so an
            // exponent form reads back the same however it was written. The
            // plain forms, which is nearly all of them, need no decode
            _ => {
                if raw.parse::<f64>().is_err() {
                    None
                } else if raw.as_bytes().iter().any(|b| *b == b'e' || *b == b'E') {
                    serde_json::from_str::<serde_json::Number>(raw)
                        .ok()
                        .map(|n| n.to_string())
                } else {
                    Some(raw.to_string())
                }
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Targeted path scan
// ---------------------------------------------------------------------------

/// Whether the text is one well formed JSON document and nothing else.
///
/// The targeted walk below skips over the parts of a document it does not
/// need, which means it would happily read a field out of text that is not
/// JSON at all. Answering a query from malformed bytes is worse than
/// answering nothing, so the document is checked first. The check allocates
/// nothing and holds its nesting on an explicit stack, so a deeply nested
/// document costs memory rather than stack frames.
fn json_is_valid(bytes: &[u8]) -> bool {
    match validate_value(bytes, skip_ws(bytes, 0)) {
        Some(end) => skip_ws(bytes, end) == bytes.len(),
        None => false,
    }
}

/// What a container is waiting for, so the walk knows which token is legal
/// next without a recursive call per level
#[derive(Clone, Copy, PartialEq, Eq)]
enum Await {
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectCommaOrEnd,
    ArrayValue,
    ArrayCommaOrEnd,
}

/// Byte offset just past the value starting at `at`, or None when the text
/// from there is not a well formed value
fn validate_value(bytes: &[u8], at: usize) -> Option<usize> {
    let mut stack: Vec<Await> = Vec::new();
    let mut i = at;
    loop {
        i = skip_ws(bytes, i);
        let state = stack.last().copied();
        match state {
            // Between elements of a container, decide what is legal here
            Some(Await::ObjectCommaOrEnd) | Some(Await::ArrayCommaOrEnd) => {
                let in_object = state == Some(Await::ObjectCommaOrEnd);
                match *bytes.get(i)? {
                    b',' => {
                        *stack.last_mut()? = if in_object {
                            Await::ObjectKey
                        } else {
                            Await::ArrayValue
                        };
                        i += 1;
                        continue;
                    }
                    b'}' if in_object => {}
                    b']' if !in_object => {}
                    _ => return None,
                }
                stack.pop();
                i += 1;
                if stack.is_empty() {
                    return Some(i);
                }
                close_child(&mut stack)?;
                continue;
            }
            Some(Await::ObjectKey) => {
                // A closing brace here ends an object whose last entry took
                // a trailing comma, which is not legal, so only an empty
                // object may close and that case is handled at open
                if *bytes.get(i)? != b'"' {
                    return None;
                }
                i = skip_string_strict(bytes, i)?;
                *stack.last_mut()? = Await::ObjectColon;
                continue;
            }
            Some(Await::ObjectColon) => {
                if *bytes.get(i)? != b':' {
                    return None;
                }
                *stack.last_mut()? = Await::ObjectValue;
                i += 1;
                continue;
            }
            _ => {}
        }

        // A value is expected here, at the top level or inside a container
        match *bytes.get(i)? {
            b'{' => {
                i = skip_ws(bytes, i + 1);
                if *bytes.get(i)? == b'}' {
                    i += 1;
                    if stack.is_empty() {
                        return Some(i);
                    }
                    close_child(&mut stack)?;
                    continue;
                }
                stack.push(Await::ObjectKey);
                continue;
            }
            b'[' => {
                i = skip_ws(bytes, i + 1);
                if *bytes.get(i)? == b']' {
                    i += 1;
                    if stack.is_empty() {
                        return Some(i);
                    }
                    close_child(&mut stack)?;
                    continue;
                }
                stack.push(Await::ArrayValue);
                continue;
            }
            b'"' => i = skip_string_strict(bytes, i)?,
            b't' => i = expect_word(bytes, i, b"true")?,
            b'f' => i = expect_word(bytes, i, b"false")?,
            b'n' => i = expect_word(bytes, i, b"null")?,
            _ => i = validate_number(bytes, i)?,
        }
        if stack.is_empty() {
            return Some(i);
        }
        close_child(&mut stack)?;
    }
}

/// Marks the container on top of the stack as holding a finished child, so
/// the next token has to be a separator or the container's end
fn close_child(stack: &mut [Await]) -> Option<()> {
    let top = stack.last_mut()?;
    *top = match *top {
        Await::ObjectValue => Await::ObjectCommaOrEnd,
        Await::ArrayValue => Await::ArrayCommaOrEnd,
        _ => return None,
    };
    Some(())
}

fn expect_word(bytes: &[u8], at: usize, word: &[u8]) -> Option<usize> {
    if bytes.len() >= at + word.len() && &bytes[at..at + word.len()] == word {
        Some(at + word.len())
    } else {
        None
    }
}

/// Byte offset just past a JSON string, rejecting the escapes and control
/// characters the grammar does not allow
fn skip_string_strict(bytes: &[u8], at: usize) -> Option<usize> {
    let mut i = at + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(i + 1),
            b'\\' => match *bytes.get(i + 1)? {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                b'u' => {
                    let digits = bytes.get(i + 2..i + 6)?;
                    if !digits.iter().all(|d| d.is_ascii_hexdigit()) {
                        return None;
                    }
                    i += 6;
                }
                _ => return None,
            },
            // A raw control character is not legal inside a string
            0x00..=0x1F => return None,
            _ => i += 1,
        }
    }
    None
}

/// Byte offset just past a JSON number, held to the grammar rather than to
/// whatever a float parser happens to accept
fn validate_number(bytes: &[u8], at: usize) -> Option<usize> {
    let mut i = at;
    if bytes.get(i) == Some(&b'-') {
        i += 1;
    }
    match *bytes.get(i)? {
        b'0' => i += 1,
        b'1'..=b'9' => {
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        _ => return None,
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        if !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            return None;
        }
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        if !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            return None;
        }
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    Some(i)
}

/// Byte offset just past a JSON string that starts at `at`, which must be its
/// opening quote. Returns None for a string that never closes
fn skip_string(bytes: &[u8], at: usize) -> Option<usize> {
    let mut i = at + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Byte offset just past the value starting at `at`.
///
/// Structured values are skipped by counting brackets while stepping over
/// strings whole, so a brace inside a string never moves the depth. Scalars
/// run to the first delimiter. Nothing here allocates, which is the point:
/// reading one field of a document should not cost a copy of the rest of it
fn skip_value(bytes: &[u8], at: usize) -> Option<usize> {
    let mut i = skip_ws(bytes, at);
    match *bytes.get(i)? {
        b'"' => skip_string(bytes, i),
        b'{' | b'[' => {
            let mut depth = 0i32;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        i = skip_string(bytes, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            while i < bytes.len()
                && !matches!(bytes[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
            {
                i += 1;
            }
            Some(i)
        }
    }
}

fn skip_ws(bytes: &[u8], mut at: usize) -> usize {
    while at < bytes.len() && matches!(bytes[at], b' ' | b'\t' | b'\n' | b'\r') {
        at += 1;
    }
    at
}

/// Whether the JSON string starting at `at` decodes to `wanted`.
///
/// Comparison runs over the encoded bytes and decodes only the escapes it
/// meets, so an ordinary key costs a memcmp and an escaped one still
/// compares by value rather than by spelling
fn key_matches(bytes: &[u8], at: usize, wanted: &str) -> bool {
    let want = wanted.as_bytes();
    let mut i = at + 1;
    let mut w = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return w == want.len(),
            b'\\' => {
                let escape = *match bytes.get(i + 1) {
                    Some(b) => b,
                    None => return false,
                };
                let decoded = match escape {
                    b'"' => b'"',
                    b'\\' => b'\\',
                    b'/' => b'/',
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'b' => 8,
                    b'f' => 12,
                    // A \u escape decodes to more than one byte, so the
                    // comparison hands the key to the full decoder rather
                    // than guessing at it
                    _ => return decode_json_string(bytes, at).as_deref() == Some(wanted),
                };
                if want.get(w) != Some(&decoded) {
                    return false;
                }
                w += 1;
                i += 2;
            }
            b => {
                if want.get(w) != Some(&b) {
                    return false;
                }
                w += 1;
                i += 1;
            }
        }
    }
    false
}

/// The raw slice of the value at a dotted path, without copying anything on
/// the way to it.
///
/// Each segment scans its object's keys and steps over the values it does
/// not want, so the cost follows the part of the document that leads to the
/// field rather than the whole of it. A path that leaves an object, meets a
/// missing key, or runs off malformed text answers None, which reads as SQL
/// NULL the same way a missing key already did
pub fn extract_raw<'a>(json_text: &'a str, dotted_path: &str) -> Option<&'a str> {
    if dotted_path.is_empty() {
        return None;
    }
    // Text that is not JSON has no field to read, and only a real decoder
    // can say so. This one validates without building anything, so the walk
    // below still costs the path it takes rather than a tree of the whole
    // document
    if !json_is_valid(json_text.as_bytes()) {
        return None;
    }
    let bytes = json_text.as_bytes();
    let mut at = skip_ws(bytes, 0);
    for segment in dotted_path.split('.') {
        if *bytes.get(at)? != b'{' {
            return None;
        }
        at = skip_ws(bytes, at + 1);
        let mut found = None;
        while at < bytes.len() && bytes[at] != b'}' {
            if bytes[at] != b'"' {
                return None;
            }
            let is_match = key_matches(bytes, at, segment);
            let key_end = skip_string(bytes, at)?;
            let colon = skip_ws(bytes, key_end);
            if *bytes.get(colon)? != b':' {
                return None;
            }
            let value_start = skip_ws(bytes, colon + 1);
            let value_end = skip_value(bytes, value_start)?;
            if is_match {
                // A document may name one key twice, and the JSON decoder
                // reads the last of them, so the walk keeps going rather
                // than stopping at the first and answering differently
                found = Some(value_start);
            }
            at = skip_ws(bytes, value_end);
            if at < bytes.len() && bytes[at] == b',' {
                at = skip_ws(bytes, at + 1);
            }
        }
        at = found?;
    }
    let end = skip_value(bytes, at)?;
    json_text.get(at..end)
}

/// Decodes the JSON string starting at `at`, resolving escapes
fn decode_json_string(bytes: &[u8], at: usize) -> Option<String> {
    let end = skip_string(bytes, at)?;
    let raw = std::str::from_utf8(&bytes[at..end]).ok()?;
    serde_json::from_str::<String>(raw).ok()
}

/// Drops all tracked state of one column, for DROP TABLE and DROP COLUMN
pub fn clear_column(table_id: u32, column_id: u16) {
    let _ = tracker().remove_sync(&(table_id, column_id));
}

/// Serializes every column's counters as a versioned JSON envelope
pub fn snapshot() -> Vec<u8> {
    let mut columns = Vec::new();
    tracker().iter_sync(|key, column| {
        let mut paths = Vec::new();
        column.paths.iter_sync(|path, counter| {
            paths.push(serde_json::json!({
                "path": path,
                "occurrences": counter.occurrences.load(Ordering::Relaxed),
                "string_count": counter.string_count.load(Ordering::Relaxed),
                "number_count": counter.number_count.load(Ordering::Relaxed),
                "bool_count": counter.bool_count.load(Ordering::Relaxed),
                "shredded": counter.shredded.load(Ordering::Relaxed),
            }));
            true
        });
        columns.push(serde_json::json!({
            "table_id": key.0,
            "column_id": key.1,
            "total_writes": column.total_writes.load(Ordering::Relaxed),
            "malformed_writes": column.malformed_writes.load(Ordering::Relaxed),
            "paths": paths,
        }));
        true
    });
    let envelope = serde_json::json!({
        "version": SNAPSHOT_VERSION,
        "columns": columns,
    });
    envelope.to_string().into_bytes()
}

fn required_u64(entry: &serde_json::Value, field: &str) -> Result<u64> {
    entry.get(field).and_then(|v| v.as_u64()).ok_or_else(|| {
        ZyronError::DecodingFailed(format!(
            "variant shred snapshot entry is missing numeric field {field}"
        ))
    })
}

/// Reloads counters from a snapshot, replacing state only for the columns
/// the snapshot holds. Any version other than the current one is rejected
pub fn restore(bytes: &[u8]) -> Result<()> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
        ZyronError::DecodingFailed(format!("variant shred snapshot is not valid json, {e}"))
    })?;
    let version = required_u64(&parsed, "version")?;
    if version != SNAPSHOT_VERSION {
        return Err(ZyronError::DecodingFailed(format!(
            "variant shred snapshot version {version} is not supported, expected {SNAPSHOT_VERSION}"
        )));
    }
    let columns = parsed
        .get("columns")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ZyronError::DecodingFailed("variant shred snapshot has no columns array".to_string())
        })?;
    for entry in columns {
        let table_id = u32::try_from(required_u64(entry, "table_id")?).map_err(|_| {
            ZyronError::DecodingFailed("variant shred snapshot table_id exceeds u32".to_string())
        })?;
        let column_id = u16::try_from(required_u64(entry, "column_id")?).map_err(|_| {
            ZyronError::DecodingFailed("variant shred snapshot column_id exceeds u16".to_string())
        })?;
        let path_entries = entry
            .get("paths")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ZyronError::DecodingFailed(
                    "variant shred snapshot column entry has no paths array".to_string(),
                )
            })?;
        let rebuilt = Arc::new(ColumnTracker {
            total_writes: AtomicU64::new(required_u64(entry, "total_writes")?),
            malformed_writes: AtomicU64::new(required_u64(entry, "malformed_writes")?),
            paths: scc::HashMap::new(),
            tracked_paths: AtomicUsize::new(0),
        });
        for path_entry in path_entries {
            let path = path_entry
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ZyronError::DecodingFailed(
                        "variant shred snapshot path entry has no path field".to_string(),
                    )
                })?;
            let counter = PathCounter {
                occurrences: AtomicU64::new(required_u64(path_entry, "occurrences")?),
                string_count: AtomicU64::new(required_u64(path_entry, "string_count")?),
                number_count: AtomicU64::new(required_u64(path_entry, "number_count")?),
                bool_count: AtomicU64::new(required_u64(path_entry, "bool_count")?),
                shredded: AtomicBool::new(
                    path_entry
                        .get("shredded")
                        .and_then(|v| v.as_bool())
                        .ok_or_else(|| {
                            ZyronError::DecodingFailed(
                                "variant shred snapshot path entry has no shredded field"
                                    .to_string(),
                            )
                        })?,
                ),
            };
            if rebuilt
                .paths
                .insert_sync(path.to_string(), Arc::new(counter))
                .is_ok()
            {
                rebuilt.tracked_paths.fetch_add(1, Ordering::Relaxed);
            }
        }
        let _ = tracker().upsert_sync((table_id, column_id), rebuilt);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Table ids at or above 900000 so tracker state never collides with
    // other tests sharing the process wide map

    fn record_many(table_id: u32, column_id: u16, json_text: &str, count: usize) {
        for _ in 0..count {
            record_variant_write(table_id, column_id, json_text);
        }
    }

    fn stats_for<'a>(stats: &'a [PathStats], path: &str) -> &'a PathStats {
        stats
            .iter()
            .find(|s| s.path == path)
            .unwrap_or_else(|| panic!("path {path} not tracked"))
    }

    #[test]
    fn test_variant_coverage_and_occurrences() {
        let table = 900_100;
        record_many(table, 0, r#"{"address":{"city":"x"},"age":1}"#, 1500);
        record_many(table, 0, r#"{"age":2}"#, 500);
        let stats = paths_for(table, 0);
        let age = stats_for(&stats, "age");
        assert_eq!(age.occurrences, 2000);
        assert!((age.coverage_percent - 100.0).abs() < 1e-9);
        assert_eq!(age.value_kind, "number");
        let city = stats_for(&stats, "address.city");
        assert_eq!(city.occurrences, 1500);
        assert!((city.coverage_percent - 75.0).abs() < 1e-9);
        assert_eq!(city.value_kind, "string");
        assert_eq!(stats[0].path, "age");
    }

    #[test]
    fn test_variant_promotion_candidates_and_mark_shredded() {
        let table = 900_101;
        record_many(table, 0, r#"{"address":{"city":"x"},"age":1}"#, 1500);
        record_many(table, 0, r#"{"age":2}"#, 500);
        let candidates = promotion_candidates(table, 0, 1000, 70.0);
        let names: Vec<&str> = candidates.iter().map(|c| c.path.as_str()).collect();
        assert_eq!(names, vec!["age", "address.city"]);

        mark_shredded(table, 0, "age");
        let remaining = promotion_candidates(table, 0, 1000, 70.0);
        let names: Vec<&str> = remaining.iter().map(|c| c.path.as_str()).collect();
        assert_eq!(names, vec!["address.city"]);
        assert_eq!(shredded_paths(table, 0), vec!["age".to_string()]);
        assert!(stats_for(&paths_for(table, 0), "age").shredded);
    }

    #[test]
    fn test_variant_mark_shredded_unseen_path() {
        let table = 900_102;
        mark_shredded(table, 0, "meta.source");
        assert_eq!(shredded_paths(table, 0), vec!["meta.source".to_string()]);
        let stats = paths_for(table, 0);
        let entry = stats_for(&stats, "meta.source");
        assert!(entry.shredded);
        assert_eq!(entry.occurrences, 0);
    }

    /// The validator decides which documents have a field to read at all,
    /// so it must accept and reject exactly what the JSON decoder does.
    /// Too lenient answers queries from corrupt text, too strict turns
    /// stored rows into nulls
    #[test]
    fn test_validator_agrees_with_the_json_decoder() {
        let cases = [
            "{}",
            "[]",
            "null",
            "true",
            "false",
            "0",
            "-0",
            "12",
            "-12.5e3",
            "1E+2",
            "1e-2",
            "\"s\"",
            r#""\u00e9""#,
            r#""\n\t\\""#,
            "{\"a\":1}",
            "{\"a\":[1,{\"b\":2}]}",
            "  {\"a\" : 1}  ",
            "{\"a\":1,\"a\":2}",
            // malformed
            "{",
            "}",
            "{\"a\":1",
            "{\"a\"1}",
            "{a:1}",
            "{\"a\":}",
            "{\"a\":1,}",
            "[1,]",
            "[1 2]",
            "01",
            "1.",
            ".5",
            "+1",
            "1e",
            "tru",
            "\"unterminated",
            r#""bad\escape""#,
            r#""\u12""#,
            "{} {}",
            "",
            "   ",
            "nan",
            "Infinity",
        ];
        for case in cases {
            let mine = json_is_valid(case.as_bytes());
            let theirs = serde_json::from_str::<serde_json::Value>(case).is_ok();
            assert_eq!(mine, theirs, "validator disagreed on {case:?}");
        }
    }

    /// The tree walk the scanner replaced, kept as the reference the
    /// differential test compares against
    fn reference_extract(json_text: &str, dotted_path: &str) -> Option<serde_json::Value> {
        if dotted_path.is_empty() {
            return None;
        }
        let parsed: serde_json::Value = serde_json::from_str(json_text).ok()?;
        let mut current = &parsed;
        for segment in dotted_path.split('.') {
            match current {
                serde_json::Value::Object(fields) => current = fields.get(segment)?,
                _ => return None,
            }
        }
        Some(current.clone())
    }

    fn reference_scalar_text(json_text: &str, dotted_path: &str) -> Option<String> {
        match reference_extract(json_text, dotted_path)? {
            serde_json::Value::String(text) => Some(text),
            serde_json::Value::Number(number) => Some(number.to_string()),
            serde_json::Value::Bool(flag) => Some(if flag { "true" } else { "false" }.to_string()),
            _ => None,
        }
    }

    /// Every documented shape, read both ways. The scanner exists to avoid
    /// building a tree, so what it must not do is answer differently from
    /// the tree it replaced
    #[test]
    fn test_scanner_agrees_with_the_tree_walk() {
        let documents = [
            r#"{"a":1,"b":"two","c":true,"d":null}"#,
            r#"{"a":{"b":{"c":"deep"}},"z":9}"#,
            r#"{"nested":{"arr":[1,2,{"x":5}],"after":"kept"}}"#,
            r#"{"esc":"a\"b\\c\nd","plain":"e"}"#,
            r#"{"unicode":"é中","tail":3}"#,
            r#"{"num":-12.5e3,"int":0,"big":9007199254740993}"#,
            r#"  {  "spaced"  :  {  "inner"  :  "v"  }  }  "#,
            r#"{"brace_in_string":"}{[]","after":1}"#,
            r#"{"empty_obj":{},"empty_arr":[],"s":""}"#,
            r#"{"dup":1,"dup":2}"#,
            r#"{"a":1"#,
            r#"not json at all"#,
            r#"[1,2,3]"#,
            r#"{}"#,
        ];
        let paths = [
            "a",
            "b",
            "c",
            "d",
            "z",
            "a.b",
            "a.b.c",
            "nested.arr",
            "nested.after",
            "esc",
            "plain",
            "unicode",
            "tail",
            "num",
            "int",
            "big",
            "spaced",
            "spaced.inner",
            "brace_in_string",
            "after",
            "empty_obj",
            "empty_arr",
            "s",
            "dup",
            "missing",
            "a.missing",
            "b.c",
            "",
        ];
        for doc in documents {
            for path in paths {
                let scanned = extract_path(doc, path);
                let reference = reference_extract(doc, path);
                assert_eq!(
                    scanned, reference,
                    "extract_path disagreed on {path} of {doc}"
                );
                let scanned_text = extract_scalar_text(doc, path);
                let reference_text = reference_scalar_text(doc, path);
                assert_eq!(
                    scanned_text, reference_text,
                    "extract_scalar_text disagreed on {path} of {doc}"
                );
            }
        }
    }

    /// Generated documents, so agreement is shown over shapes nobody wrote
    /// by hand
    #[test]
    fn test_scanner_agrees_over_generated_documents() {
        let mut seed = 0x5EED_1234u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 11
        };
        let names = ["alpha", "beta", "gamma", "delta", "eps"];
        for _ in 0..400 {
            let mut doc = String::from("{");
            let mut expected_paths = Vec::new();
            let field_count = 1 + (next() % 5) as usize;
            for f in 0..field_count {
                if f > 0 {
                    doc.push(',');
                }
                let name = names[(next() as usize) % names.len()];
                expected_paths.push(name.to_string());
                let shape = next() % 6;
                match shape {
                    0 => doc.push_str(&format!("\"{name}\":{}", next() % 1000)),
                    1 => doc.push_str(&format!("\"{name}\":\"v{}\"", next() % 100)),
                    2 => doc.push_str(&format!("\"{name}\":{}", next() % 2 == 0)),
                    3 => doc.push_str(&format!("\"{name}\":null")),
                    4 => {
                        let inner = names[(next() as usize) % names.len()];
                        expected_paths.push(format!("{name}.{inner}"));
                        doc.push_str(&format!("\"{name}\":{{\"{inner}\":{}}}", next() % 50));
                    }
                    _ => doc.push_str(&format!("\"{name}\":[{},{}]", next() % 10, next() % 10)),
                }
            }
            doc.push('}');
            for path in &expected_paths {
                assert_eq!(
                    extract_path(&doc, path),
                    reference_extract(&doc, path),
                    "extract_path disagreed on {path} of {doc}"
                );
                assert_eq!(
                    extract_scalar_text(&doc, path),
                    reference_scalar_text(&doc, path),
                    "extract_scalar_text disagreed on {path} of {doc}"
                );
            }
            assert_eq!(extract_path(&doc, "absent"), None);
        }
    }

    /// Reading one field must not depend on the size of the rest of the
    /// document, which is the whole reason the scanner exists
    #[test]
    fn test_scanner_skips_the_rest_of_the_document() {
        let filler: String = (0..500)
            .map(|i| format!("\"pad{i}\":\"{}\",", "x".repeat(40)))
            .collect();
        let doc = format!("{{\"wanted\":\"here\",{filler}\"last\":1}}");
        assert_eq!(
            extract_scalar_text(&doc, "wanted"),
            Some("here".to_string())
        );
        assert_eq!(extract_scalar_text(&doc, "last"), Some("1".to_string()));
        assert_eq!(
            extract_path(&doc, "wanted"),
            reference_extract(&doc, "wanted")
        );
    }

    #[test]
    fn test_variant_extract_path() {
        let json = r#"{"address":{"city":"x","geo":{"lat":1.5}},"age":1}"#;
        assert_eq!(
            extract_path(json, "address.city"),
            Some(serde_json::Value::String("x".to_string()))
        );
        assert!(extract_path(json, "address.geo").is_some());
        assert_eq!(extract_path(json, "address.zip"), None);
        assert_eq!(extract_path(json, "age.years"), None);
        assert_eq!(extract_path(json, ""), None);
        assert_eq!(extract_path("not json", "age"), None);
    }

    #[test]
    fn test_variant_extract_scalar_text() {
        let json =
            r#"{"address":{"city":"x"},"age":1,"score":2.5,"active":true,"tags":[1],"note":null}"#;
        assert_eq!(
            extract_scalar_text(json, "address.city"),
            Some("x".to_string())
        );
        assert_eq!(extract_scalar_text(json, "age"), Some("1".to_string()));
        assert_eq!(extract_scalar_text(json, "score"), Some("2.5".to_string()));
        assert_eq!(
            extract_scalar_text(json, "active"),
            Some("true".to_string())
        );
        assert_eq!(extract_scalar_text(json, "address"), None);
        assert_eq!(extract_scalar_text(json, "tags"), None);
        assert_eq!(extract_scalar_text(json, "note"), None);
        assert_eq!(extract_scalar_text(json, "missing"), None);
    }

    #[test]
    fn test_variant_malformed_json_counts_without_panicking() {
        let table = 900_104;
        record_variant_write(table, 0, "{not json");
        record_variant_write(table, 0, "");
        record_variant_write(table, 0, r#"{"ok":1}"#);
        assert!(paths_for(table, 0).iter().any(|s| s.path == "ok"));

        let parsed: serde_json::Value = match serde_json::from_slice(&snapshot()) {
            Ok(value) => value,
            Err(e) => panic!("snapshot is not valid json, {e}"),
        };
        let column = parsed["columns"]
            .as_array()
            .and_then(|cols| {
                cols.iter()
                    .find(|c| c["table_id"].as_u64() == Some(u64::from(table)))
            })
            .unwrap_or_else(|| panic!("snapshot has no column for table {table}"));
        assert_eq!(column["malformed_writes"].as_u64(), Some(2));
        assert_eq!(column["total_writes"].as_u64(), Some(1));
    }

    #[test]
    fn test_variant_depth_capped_at_four_segments() {
        let table = 900_105;
        record_variant_write(table, 0, r#"{"a":{"b":{"c":{"d":1,"e":{"f":2}}}}}"#);
        let paths: Vec<String> = paths_for(table, 0).into_iter().map(|s| s.path).collect();
        assert_eq!(paths, vec!["a.b.c.d".to_string()]);
    }

    #[test]
    fn test_variant_dotted_keys_skipped() {
        let table = 900_106;
        record_variant_write(table, 0, r#"{"a.b":1,"c":2}"#);
        let paths: Vec<String> = paths_for(table, 0).into_iter().map(|s| s.path).collect();
        assert_eq!(paths, vec!["c".to_string()]);
    }

    #[test]
    fn test_variant_top_level_non_objects_bump_only_total() {
        let table = 900_107;
        record_variant_write(table, 0, r#"{"k":1}"#);
        record_variant_write(table, 0, "[1,2,3]");
        record_variant_write(table, 0, "42");
        record_variant_write(table, 0, "null");
        let stats = paths_for(table, 0);
        let k = stats_for(&stats, "k");
        assert_eq!(k.occurrences, 1);
        assert!((k.coverage_percent - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_variant_path_cap_at_256() {
        let table = 900_108;
        for i in 0..300 {
            record_variant_write(table, 0, &format!("{{\"k{i:03}\":1}}"));
        }
        let stats = paths_for(table, 0);
        assert_eq!(stats.len(), 256);
        assert!((stats[0].coverage_percent - (100.0 / 300.0)).abs() < 1e-9);
    }

    #[test]
    fn test_variant_snapshot_restore_round_trip_and_clear() {
        let table = 900_109;
        record_many(table, 3, r#"{"address":{"city":"x"},"age":1}"#, 30);
        record_many(table, 3, r#"{"age":2}"#, 10);
        record_variant_write(table, 3, "{broken");
        mark_shredded(table, 3, "age");

        // The snapshot holds every column in the process, so the restore
        // input is narrowed to this test's table to leave columns owned by
        // concurrently running tests untouched
        let full: serde_json::Value = match serde_json::from_slice(&snapshot()) {
            Ok(value) => value,
            Err(e) => panic!("snapshot is not valid json, {e}"),
        };
        let ours: Vec<serde_json::Value> = full["columns"]
            .as_array()
            .map(|cols| {
                cols.iter()
                    .filter(|c| c["table_id"].as_u64() == Some(u64::from(table)))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(ours.len(), 1);
        let narrowed = serde_json::json!({"version": 1, "columns": ours})
            .to_string()
            .into_bytes();

        clear_column(table, 3);
        assert!(paths_for(table, 3).is_empty());

        match restore(&narrowed) {
            Ok(()) => {}
            Err(e) => panic!("restore failed, {e:?}"),
        }
        let stats = paths_for(table, 3);
        let age = stats_for(&stats, "age");
        assert_eq!(age.occurrences, 40);
        assert!((age.coverage_percent - 100.0).abs() < 1e-9);
        assert!(age.shredded);
        assert_eq!(age.value_kind, "number");
        let city = stats_for(&stats, "address.city");
        assert_eq!(city.occurrences, 30);
        assert!((city.coverage_percent - 75.0).abs() < 1e-9);
        assert!(!city.shredded);
        assert_eq!(shredded_paths(table, 3), vec!["age".to_string()]);
    }

    #[test]
    fn test_variant_restore_rejects_other_versions_and_garbage() {
        let unsupported = restore(br#"{"version":2,"columns":[]}"#);
        match unsupported {
            Err(ZyronError::DecodingFailed(message)) => {
                assert!(message.contains("version 2"));
            }
            other => panic!("expected DecodingFailed, got {other:?}"),
        }
        assert!(restore(b"not a snapshot").is_err());
        assert!(restore(br#"{"columns":[]}"#).is_err());
    }

    #[test]
    fn test_variant_clear_column_scoped_to_one_column() {
        let table = 900_110;
        record_variant_write(table, 0, r#"{"a":1}"#);
        record_variant_write(table, 1, r#"{"b":1}"#);
        clear_column(table, 0);
        assert!(paths_for(table, 0).is_empty());
        assert_eq!(paths_for(table, 1).len(), 1);
    }
}
