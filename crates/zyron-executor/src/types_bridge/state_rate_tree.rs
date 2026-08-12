//! state_machine, rate_limit, and hierarchy dispatch arms
//!
//! opaque values travel as JSON text bytes inside Binary cells
//! a state machine def round-trips as the canonical JSON definition sm_parse accepts,
//! StateMachineDef has no Serialize impl and no byte codec of its own
//! a bucket round-trips as a 4 element JSON array of current, capacity, rate, last_micros,
//! consume and add wrap the updated bucket in an object that also carries the
//! allowed or accepted flag, the bucket reader accepts both forms so results chain
//! Array typed results are JSON arrays per the types bridge contract

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_types::{hierarchy, rate_limit, state_machine};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "sm_parse" => sm_parse_col(args),
        "sm_transition" => sm_transition_col(args),
        "sm_can_transition" => sm_can_transition_col(args),
        "sm_is_terminal" => sm_is_terminal_col(args),
        "sm_available_events" => sm_available_events_col(args),
        "sm_reachable_states" => sm_reachable_states_col(args),
        "sm_shortest_path" => sm_shortest_path_col(args),
        "token_bucket_create" => token_bucket_create_col(args),
        "token_bucket_consume" => token_bucket_consume_col(args),
        "token_bucket_available" => token_bucket_available_col(args),
        "leaky_bucket_create" => leaky_bucket_create_col(args),
        "leaky_bucket_add" => leaky_bucket_add_col(args),
        "sliding_window_count" => sliding_window_count_col(args),
        "sliding_window_check" => sliding_window_check_col(args),
        "fixed_window_count" => fixed_window_count_col(args),
        "closure_table_ancestors" => closure_table_ancestors_col(args),
        "closure_table_descendants" => closure_table_descendants_col(args),
        "closure_table_depth" => closure_table_depth_col(args),
        "closure_table_insert" => closure_table_insert_col(args),
        "is_ancestor" => is_ancestor_col(args),
        "materialized_path" => materialized_path_col(args),
        "nested_set_rebuild" => nested_set_rebuild_col(args),
        "nested_set_subtree" => nested_set_subtree_col(args),
        "path_ancestors" => path_ancestors_col(args),
        "path_depth" => path_depth_col(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// state_machine
// ---------------------------------------------------------------------------

// sm_parse(text) -> bytea, an unparseable definition yields NULL
fn sm_parse_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_parse(text)";
    arg_check(args, 1, sig)?;
    let defs = text_column(&args[0], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Bytea, |i| {
        let def = parse_def(defs[i]?)?;
        Some(def_to_bytes(&def))
    }))
}

// unknown state or missing transition yields NULL, sm_can_transition tests validity
fn sm_transition_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_transition(statemachine, text, text)";
    arg_check(args, 3, sig)?;
    let defs = text_column(&args[0], sig)?;
    let states = text_column(&args[1], sig)?;
    let events = text_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_utf8(args, n, |i| {
        let def = parse_def(defs[i]?)?;
        state_machine::sm_transition(&def, states[i]?, events[i]?).ok()
    }))
}

fn sm_can_transition_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_can_transition(statemachine, text, text)";
    arg_check(args, 3, sig)?;
    let defs = text_column(&args[0], sig)?;
    let states = text_column(&args[1], sig)?;
    let events = text_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_bool(args, n, |i| {
        let def = parse_def(defs[i]?)?;
        Some(state_machine::sm_can_transition(
            &def, states[i]?, events[i]?,
        ))
    }))
}

fn sm_is_terminal_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_is_terminal(statemachine, text)";
    arg_check(args, 2, sig)?;
    let defs = text_column(&args[0], sig)?;
    let states = text_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_bool(args, n, |i| {
        let def = parse_def(defs[i]?)?;
        Some(state_machine::sm_is_terminal(&def, states[i]?))
    }))
}

// set iteration order is unspecified, output sorted for determinism
fn sm_available_events_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_available_events(statemachine, text)";
    arg_check(args, 2, sig)?;
    let defs = text_column(&args[0], sig)?;
    let states = text_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let def = parse_def(defs[i]?)?;
        let mut events = state_machine::sm_available_events(&def, states[i]?);
        events.sort();
        Some(json_string_array(&events))
    }))
}

// set iteration order is unspecified, output sorted for determinism
fn sm_reachable_states_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_reachable_states(statemachine, text)";
    arg_check(args, 2, sig)?;
    let defs = text_column(&args[0], sig)?;
    let states = text_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let def = parse_def(defs[i]?)?;
        let mut reachable = state_machine::sm_reachable_states(&def, states[i]?);
        reachable.sort();
        Some(json_string_array(&reachable))
    }))
}

// unreachable target yields NULL, an empty array means already at the target
fn sm_shortest_path_col(args: &[Column]) -> Result<Column> {
    let sig = "sm_shortest_path(statemachine, text, text)";
    arg_check(args, 3, sig)?;
    let defs = text_column(&args[0], sig)?;
    let froms = text_column(&args[1], sig)?;
    let tos = text_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let def = parse_def(defs[i]?)?;
        let path = state_machine::sm_shortest_path(&def, froms[i]?, tos[i]?)?;
        Some(json_string_array(&path))
    }))
}

// ---------------------------------------------------------------------------
// rate_limit
// ---------------------------------------------------------------------------

fn token_bucket_create_col(args: &[Column]) -> Result<Column> {
    let sig = "token_bucket_create(float, float)";
    arg_check(args, 2, sig)?;
    let caps = float_column(&args[0], sig)?;
    let rates = float_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Bytea, |i| {
        Some(bucket_bytes(rate_limit::token_bucket_create(
            caps[i], rates[i],
        )))
    }))
}

fn token_bucket_consume_col(args: &[Column]) -> Result<Column> {
    let sig = "token_bucket_consume(bucket, float, bigint)";
    arg_check(args, 3, sig)?;
    let buckets = text_column(&args[0], sig)?;
    let tokens = float_column(&args[1], sig)?;
    let nows = int_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Bytea, |i| {
        let bucket = parse_bucket(buckets[i]?)?;
        let (allowed, updated) = rate_limit::token_bucket_consume(bucket, tokens[i], nows[i]);
        Some(bucket_flag_bytes("allowed", allowed, updated))
    }))
}

fn token_bucket_available_col(args: &[Column]) -> Result<Column> {
    let sig = "token_bucket_available(bucket, bigint)";
    arg_check(args, 2, sig)?;
    let buckets = text_column(&args[0], sig)?;
    let nows = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_f64(args, n, |i| {
        let bucket = parse_bucket(buckets[i]?)?;
        Some(rate_limit::token_bucket_available(bucket, nows[i]))
    }))
}

fn leaky_bucket_create_col(args: &[Column]) -> Result<Column> {
    let sig = "leaky_bucket_create(float, float)";
    arg_check(args, 2, sig)?;
    let caps = float_column(&args[0], sig)?;
    let rates = float_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Bytea, |i| {
        Some(bucket_bytes(rate_limit::leaky_bucket_create(
            caps[i], rates[i],
        )))
    }))
}

fn leaky_bucket_add_col(args: &[Column]) -> Result<Column> {
    let sig = "leaky_bucket_add(bucket, float, bigint)";
    arg_check(args, 3, sig)?;
    let buckets = text_column(&args[0], sig)?;
    let amounts = float_column(&args[1], sig)?;
    let nows = int_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Bytea, |i| {
        let bucket = parse_bucket(buckets[i]?)?;
        let (accepted, updated) = rate_limit::leaky_bucket_add(bucket, amounts[i], nows[i]);
        Some(bucket_flag_bytes("accepted", accepted, updated))
    }))
}

// timestamps sorted before the binary search the count relies on
fn sliding_window_count_col(args: &[Column]) -> Result<Column> {
    let sig = "sliding_window_count(array, bigint, bigint)";
    arg_check(args, 3, sig)?;
    let series = text_column(&args[0], sig)?;
    let windows = int_column(&args[1], sig)?;
    let nows = int_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_i64(args, n, |i| {
        let mut ts = parse_i64_array(series[i]?)?;
        ts.sort_unstable();
        Some(rate_limit::sliding_window_count(&ts, windows[i], nows[i]) as i64)
    }))
}

fn sliding_window_check_col(args: &[Column]) -> Result<Column> {
    let sig = "sliding_window_check(array, bigint, bigint, bigint)";
    arg_check(args, 4, sig)?;
    let series = text_column(&args[0], sig)?;
    let windows = int_column(&args[1], sig)?;
    let maxes = int_column(&args[2], sig)?;
    let nows = int_column(&args[3], sig)?;
    let n = row_count(args);
    Ok(nullable_bool(args, n, |i| {
        let mut ts = parse_i64_array(series[i]?)?;
        ts.sort_unstable();
        let max_count = usize::try_from(maxes[i].max(0)).unwrap_or(0);
        Some(rate_limit::sliding_window_check(
            &ts, windows[i], max_count, nows[i],
        ))
    }))
}

fn fixed_window_count_col(args: &[Column]) -> Result<Column> {
    let sig = "fixed_window_count(bigint, bigint)";
    arg_check(args, 2, sig)?;
    let stamps = int_column(&args[0], sig)?;
    let windows = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_i64(args, n, |i| {
        Some(rate_limit::fixed_window_count(stamps[i], windows[i]))
    }))
}

// ---------------------------------------------------------------------------
// hierarchy
// ---------------------------------------------------------------------------

fn closure_table_ancestors_col(args: &[Column]) -> Result<Column> {
    let sig = "closure_table_ancestors(array, bigint)";
    arg_check(args, 2, sig)?;
    let closures = text_column(&args[0], sig)?;
    let nodes = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let closure = parse_closure(closures[i]?)?;
        Some(json_i64_array(&hierarchy::closure_table_ancestors(
            &closure, nodes[i],
        )))
    }))
}

fn closure_table_descendants_col(args: &[Column]) -> Result<Column> {
    let sig = "closure_table_descendants(array, bigint)";
    arg_check(args, 2, sig)?;
    let closures = text_column(&args[0], sig)?;
    let nodes = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let closure = parse_closure(closures[i]?)?;
        Some(json_i64_array(&hierarchy::closure_table_descendants(
            &closure, nodes[i],
        )))
    }))
}

fn closure_table_depth_col(args: &[Column]) -> Result<Column> {
    let sig = "closure_table_depth(array, bigint)";
    arg_check(args, 2, sig)?;
    let closures = text_column(&args[0], sig)?;
    let nodes = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_i32(args, n, |i| {
        let closure = parse_closure(closures[i]?)?;
        Some(hierarchy::closure_table_depth(&closure, nodes[i]))
    }))
}

fn closure_table_insert_col(args: &[Column]) -> Result<Column> {
    let sig = "closure_table_insert(array, bigint, bigint)";
    arg_check(args, 3, sig)?;
    let closures = text_column(&args[0], sig)?;
    let parents = int_column(&args[1], sig)?;
    let children = int_column(&args[2], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let closure = parse_closure(closures[i]?)?;
        let rows = hierarchy::closure_table_insert(&closure, parents[i], children[i]);
        Some(json_closure_rows(&rows))
    }))
}

fn is_ancestor_col(args: &[Column]) -> Result<Column> {
    let sig = "is_ancestor(text, text)";
    arg_check(args, 2, sig)?;
    let ancestors = text_column(&args[0], sig)?;
    let descendants = text_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_bool(args, n, |i| {
        Some(hierarchy::is_ancestor(ancestors[i]?, descendants[i]?))
    }))
}

// input is a JSON array of path segment strings
fn materialized_path_col(args: &[Column]) -> Result<Column> {
    let sig = "materialized_path(array)";
    arg_check(args, 1, sig)?;
    let segments = text_column(&args[0], sig)?;
    let n = row_count(args);
    Ok(nullable_utf8(args, n, |i| {
        let segs = parse_string_array(segments[i]?)?;
        let refs: Vec<&str> = segs.iter().map(|s| s.as_str()).collect();
        Some(hierarchy::materialized_path(&refs))
    }))
}

// input is a JSON array of [id, parent] pairs, parent null marks a root
fn nested_set_rebuild_col(args: &[Column]) -> Result<Column> {
    let sig = "nested_set_rebuild(array)";
    arg_check(args, 1, sig)?;
    let relations = text_column(&args[0], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let rels = parse_parent_child(relations[i]?)?;
        Some(json_node_rows(&hierarchy::nested_set_rebuild(&rels)))
    }))
}

// input is a JSON array of [id, lft, rgt] triples
fn nested_set_subtree_col(args: &[Column]) -> Result<Column> {
    let sig = "nested_set_subtree(array, bigint)";
    arg_check(args, 2, sig)?;
    let node_sets = text_column(&args[0], sig)?;
    let nodes = int_column(&args[1], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        let set = parse_nested_nodes(node_sets[i]?)?;
        Some(json_i64_array(&hierarchy::nested_set_subtree(
            &set, nodes[i],
        )))
    }))
}

fn path_ancestors_col(args: &[Column]) -> Result<Column> {
    let sig = "path_ancestors(text)";
    arg_check(args, 1, sig)?;
    let paths = text_column(&args[0], sig)?;
    let n = row_count(args);
    Ok(nullable_binary(args, n, TypeId::Array, |i| {
        Some(json_string_array(&hierarchy::path_ancestors(paths[i]?)))
    }))
}

fn path_depth_col(args: &[Column]) -> Result<Column> {
    let sig = "path_depth(text)";
    arg_check(args, 1, sig)?;
    let paths = text_column(&args[0], sig)?;
    let n = row_count(args);
    Ok(nullable_i32(args, n, |i| {
        Some(hierarchy::path_depth(paths[i]?))
    }))
}

// ---------------------------------------------------------------------------
// value codecs
// ---------------------------------------------------------------------------

fn parse_def(s: &str) -> Option<state_machine::StateMachineDef> {
    state_machine::sm_parse(s).ok()
}

// canonical JSON definition text, the same shape sm_parse accepts, so defs round-trip
fn def_to_bytes(def: &state_machine::StateMachineDef) -> Vec<u8> {
    let transitions: Vec<serde_json::Value> = def
        .transitions
        .iter()
        .map(|t| {
            let mut m = serde_json::Map::new();
            m.insert(
                "from".to_string(),
                serde_json::Value::String(t.from_state.clone()),
            );
            m.insert(
                "event".to_string(),
                serde_json::Value::String(t.event.clone()),
            );
            m.insert(
                "to".to_string(),
                serde_json::Value::String(t.to_state.clone()),
            );
            if let Some(g) = &t.guard {
                m.insert("guard".to_string(), serde_json::Value::String(g.clone()));
            }
            serde_json::Value::Object(m)
        })
        .collect();
    let mut root = serde_json::Map::new();
    root.insert(
        "states".to_string(),
        serde_json::Value::Array(
            def.states
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    root.insert(
        "initial".to_string(),
        serde_json::Value::String(def.initial_state.clone()),
    );
    root.insert(
        "transitions".to_string(),
        serde_json::Value::Array(transitions),
    );
    serde_json::Value::Object(root).to_string().into_bytes()
}

// accepts the bare 4 element array from create or the object form from consume and add
fn parse_bucket(s: &str) -> Option<(f64, f64, f64, i64)> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let arr = match v {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(mut m) => match m.remove("bucket") {
            Some(serde_json::Value::Array(a)) => a,
            _ => return None,
        },
        _ => return None,
    };
    if arr.len() != 4 {
        return None;
    }
    Some((
        arr[0].as_f64()?,
        arr[1].as_f64()?,
        arr[2].as_f64()?,
        arr[3].as_i64()?,
    ))
}

// the tuple serializes as a JSON array
fn bucket_bytes(bucket: (f64, f64, f64, i64)) -> Vec<u8> {
    serde_json::to_string(&bucket)
        .unwrap_or_default()
        .into_bytes()
}

fn bucket_flag_bytes(flag: &str, value: bool, bucket: (f64, f64, f64, i64)) -> Vec<u8> {
    let mut obj = serde_json::Map::new();
    obj.insert(flag.to_string(), serde_json::Value::Bool(value));
    obj.insert(
        "bucket".to_string(),
        serde_json::json!([bucket.0, bucket.1, bucket.2, bucket.3]),
    );
    serde_json::Value::Object(obj).to_string().into_bytes()
}

fn parse_i64_array(s: &str) -> Option<Vec<i64>> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    v.as_array()?.iter().map(|e| e.as_i64()).collect()
}

// JSON array of fixed width integer rows
fn parse_i64_rows(s: &str, width: usize) -> Option<Vec<Vec<i64>>> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let rows = v.as_array()?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let cells = row.as_array()?;
        if cells.len() != width {
            return None;
        }
        let mut parsed = Vec::with_capacity(width);
        for cell in cells {
            parsed.push(cell.as_i64()?);
        }
        out.push(parsed);
    }
    Some(out)
}

// closure rows are [ancestor, descendant, depth]
fn parse_closure(s: &str) -> Option<Vec<(i64, i64, i32)>> {
    parse_i64_rows(s, 3)?
        .into_iter()
        .map(|r| Some((r[0], r[1], i32::try_from(r[2]).ok()?)))
        .collect()
}

// nested set rows are [id, lft, rgt]
fn parse_nested_nodes(s: &str) -> Option<Vec<(i64, i32, i32)>> {
    parse_i64_rows(s, 3)?
        .into_iter()
        .map(|r| Some((r[0], i32::try_from(r[1]).ok()?, i32::try_from(r[2]).ok()?)))
        .collect()
}

fn parse_parent_child(s: &str) -> Option<Vec<(i64, Option<i64>)>> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let rows = v.as_array()?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let cells = row.as_array()?;
        if cells.len() != 2 {
            return None;
        }
        let id = cells[0].as_i64()?;
        let parent = match &cells[1] {
            serde_json::Value::Null => None,
            other => Some(other.as_i64()?),
        };
        out.push((id, parent));
    }
    Some(out)
}

fn parse_string_array(s: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    v.as_array()?
        .iter()
        .map(|e| e.as_str().map(|x| x.to_string()))
        .collect()
}

fn json_i64_array(vals: &[i64]) -> Vec<u8> {
    serde_json::to_string(vals).unwrap_or_default().into_bytes()
}

fn json_string_array(vals: &[String]) -> Vec<u8> {
    serde_json::to_string(vals).unwrap_or_default().into_bytes()
}

fn json_closure_rows(rows: &[(i64, i64, i32)]) -> Vec<u8> {
    serde_json::to_string(rows).unwrap_or_default().into_bytes()
}

fn json_node_rows(rows: &[(i64, i32, i32)]) -> Vec<u8> {
    serde_json::to_string(rows).unwrap_or_default().into_bytes()
}

// ---------------------------------------------------------------------------
// column adapters
// ---------------------------------------------------------------------------

fn arg_check(args: &[Column], expected: usize, sig: &str) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} arguments",
            sig, expected
        )));
    }
    Ok(())
}

fn row_count(args: &[Column]) -> usize {
    args.iter().map(|c| c.data.len()).min().unwrap_or(0)
}

// per cell text view, Binary cells hold JSON or definition text, invalid utf8 reads as NULL
fn text_column<'a>(col: &'a Column, sig: &str) -> Result<Vec<Option<&'a str>>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v.iter().map(|s| Some(s.as_str())).collect()),
        ColumnData::Binary(v) => Ok(v.iter().map(|b| std::str::from_utf8(b).ok()).collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a text or binary argument",
            sig
        ))),
    }
}

fn int_column(col: &Column, sig: &str) -> Result<Vec<i64>> {
    super::column_ints(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects an integer argument", sig)))
}

fn float_column(col: &Column, sig: &str) -> Result<Vec<f64>> {
    super::column_floats(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a numeric argument", sig)))
}

fn any_null(args: &[Column], i: usize) -> bool {
    args.iter().any(|c| c.nulls.is_null(i))
}

// each nullable builder applies f per row, a NULL input or a None result yields NULL

fn nullable_utf8<F: Fn(usize) -> Option<String>>(args: &[Column], n: usize, f: F) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Utf8(data), nulls, TypeId::Varchar)
}

fn nullable_binary<F: Fn(usize) -> Option<Vec<u8>>>(
    args: &[Column],
    n: usize,
    type_id: TypeId,
    f: F,
) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Binary(data), nulls, type_id)
}

fn nullable_bool<F: Fn(usize) -> Option<bool>>(args: &[Column], n: usize, f: F) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(false);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Boolean(data), nulls, TypeId::Boolean)
}

fn nullable_i32<F: Fn(usize) -> Option<i32>>(args: &[Column], n: usize, f: F) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Int32(data), nulls, TypeId::Int32)
}

fn nullable_i64<F: Fn(usize) -> Option<i64>>(args: &[Column], n: usize, f: F) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Int64(data), nulls, TypeId::Int64)
}

fn nullable_f64<F: Fn(usize) -> Option<f64>>(args: &[Column], n: usize, f: F) -> Column {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i) };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Float64(data), nulls, TypeId::Float64)
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

    fn float_col(vals: &[f64]) -> Column {
        Column::new(ColumnData::Float64(vals.to_vec()), TypeId::Float64)
    }

    fn binary_cell(col: &Column, i: usize) -> Vec<u8> {
        match &col.data {
            ColumnData::Binary(v) => v[i].clone(),
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    const DEF: &str = r#"{"states":["pending","active","done"],"initial":"pending","transitions":[{"from":"pending","event":"start","to":"active"},{"from":"active","event":"finish","to":"done"}]}"#;

    #[test]
    fn sm_parse_roundtrips_through_transition() {
        let parsed = dispatch("sm_parse", &[utf8_col(&[DEF])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.type_id, TypeId::Bytea);
        assert!(!parsed.nulls.is_null(0));
        let out = dispatch(
            "sm_transition",
            &[parsed, utf8_col(&["pending"]), utf8_col(&["start"])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "active"),
            other => panic!("expected utf8 column, got {:?}", other),
        }
    }

    #[test]
    fn sm_shortest_path_missing_route_is_null() {
        let cols = [
            utf8_col(&[DEF]),
            utf8_col(&["done"]),
            utf8_col(&["pending"]),
        ];
        let out = dispatch("sm_shortest_path", &cols, 1).unwrap().unwrap();
        assert!(out.nulls.is_null(0));
    }

    #[test]
    fn token_bucket_consume_chains_into_available() {
        let created = dispatch(
            "token_bucket_create",
            &[float_col(&[10.0]), float_col(&[1.0])],
            1,
        )
        .unwrap()
        .unwrap();
        let consumed = dispatch(
            "token_bucket_consume",
            &[created, float_col(&[4.0]), int_col(&[0])],
            1,
        )
        .unwrap()
        .unwrap();
        let txt = String::from_utf8(binary_cell(&consumed, 0)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
        assert_eq!(v["allowed"], serde_json::Value::Bool(true));
        let avail = dispatch("token_bucket_available", &[consumed, int_col(&[0])], 1)
            .unwrap()
            .unwrap();
        match &avail.data {
            ColumnData::Float64(f) => assert!((f[0] - 6.0).abs() < 1e-9),
            other => panic!("expected float64 column, got {:?}", other),
        }
    }

    #[test]
    fn sliding_window_count_counts_recent_events() {
        let ts = utf8_col(&["[0,1000000,2000000,3000000,4000000]"]);
        let out = dispatch(
            "sliding_window_count",
            &[ts, int_col(&[2000000]), int_col(&[4000000])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Int64(v) => assert_eq!(v[0], 3),
            other => panic!("expected int64 column, got {:?}", other),
        }
    }

    #[test]
    fn closure_insert_links_child_to_ancestors() {
        let closure = utf8_col(&["[[1,1,0]]"]);
        let inserted = dispatch(
            "closure_table_insert",
            &[closure, int_col(&[1]), int_col(&[2])],
            1,
        )
        .unwrap()
        .unwrap();
        let txt = String::from_utf8(binary_cell(&inserted, 0)).unwrap();
        let rows: Vec<Vec<i64>> = serde_json::from_str(&txt).unwrap();
        assert!(rows.contains(&vec![2, 2, 0]));
        assert!(rows.contains(&vec![1, 2, 1]));
    }

    #[test]
    fn nested_set_rebuild_root_spans_tree() {
        let pairs = utf8_col(&["[[1,null],[2,1],[3,1]]"]);
        let out = dispatch("nested_set_rebuild", &[pairs], 1)
            .unwrap()
            .unwrap();
        let txt = String::from_utf8(binary_cell(&out, 0)).unwrap();
        let rows: Vec<Vec<i64>> = serde_json::from_str(&txt).unwrap();
        let root = rows.iter().find(|r| r[0] == 1).unwrap();
        assert_eq!((root[1], root[2]), (1, 6));
    }

    #[test]
    fn path_depth_propagates_null() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(0);
        let col = Column::with_nulls(
            ColumnData::Utf8(vec![String::new(), "/a/b/c".to_string()]),
            nulls,
            TypeId::Text,
        );
        let out = dispatch("path_depth", &[col], 2).unwrap().unwrap();
        assert!(out.nulls.is_null(0));
        assert!(!out.nulls.is_null(1));
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[1], 3),
            other => panic!("expected int32 column, got {:?}", other),
        }
    }

    #[test]
    fn path_ancestors_and_is_ancestor_agree() {
        let anc = dispatch("path_ancestors", &[utf8_col(&["/a/b/c"])], 1)
            .unwrap()
            .unwrap();
        let txt = String::from_utf8(binary_cell(&anc, 0)).unwrap();
        let list: Vec<String> = serde_json::from_str(&txt).unwrap();
        assert_eq!(list, vec!["/", "/a", "/a/b"]);
        let flag = dispatch(
            "is_ancestor",
            &[utf8_col(&["/a"]), utf8_col(&["/a/b/c"])],
            1,
        )
        .unwrap()
        .unwrap();
        match &flag.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected boolean column, got {:?}", other),
        }
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("not_a_fn", &[], 1).is_none());
    }
}
