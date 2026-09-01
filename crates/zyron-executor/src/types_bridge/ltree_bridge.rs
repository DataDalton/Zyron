//! ltree dispatch arms
//!
//! paths and lqueries travel as text, ltree_matches_any takes its lquery list
//! as a JSON string array, the form SQL arrays arrive in as JSON text bytes
//! inside Binary cells tagged Array
//! a NULL input row yields a NULL output row, malformed values error

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_types::ltree;

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "nlevel" => nlevel_col(args),
        "subpath" => subpath_col(args),
        "ltree_index" => ltree_index_col(args),
        "lca" => lca_col(args),
        "build_path" => build_path_col(args),
        "ltree_is_ancestor" => ltree_is_ancestor_col(args),
        "ltree_is_descendant" => ltree_is_descendant_col(args),
        "ltree_matches" => ltree_matches_col(args),
        "ltree_matches_any" => ltree_matches_any_col(args),
        _ => return None,
    })
}

fn nlevel_col(args: &[Column]) -> Result<Column> {
    let sig = "nlevel(ltree)";
    arg_check(args, 1, sig)?;
    let paths = text_column(&args[0], sig)?;
    let n = row_count(args);
    build_i64(args, n, |i| match paths[i] {
        Some(p) => ltree::nlevel(p).map(Some),
        None => Ok(None),
    })
}

// the third argument is optional, when present it is the label count to take
fn subpath_col(args: &[Column]) -> Result<Column> {
    let sig = "subpath(ltree, offset[, len])";
    if args.len() != 2 && args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes 2 or 3 arguments",
            sig
        )));
    }
    let paths = text_column(&args[0], sig)?;
    let offsets = int_column(&args[1], sig)?;
    let lens = match args.get(2) {
        Some(col) => Some(int_column(col, sig)?),
        None => None,
    };
    let n = row_count(args);
    build_utf8(args, n, |i| match paths[i] {
        Some(p) => ltree::subpath(p, offsets[i], lens.as_ref().map(|l| l[i])).map(Some),
        None => Ok(None),
    })
}

fn ltree_index_col(args: &[Column]) -> Result<Column> {
    let sig = "ltree_index(ltree, ltree)";
    arg_check(args, 2, sig)?;
    let paths = text_column(&args[0], sig)?;
    let subs = text_column(&args[1], sig)?;
    let n = row_count(args);
    build_i64(args, n, |i| match (paths[i], subs[i]) {
        (Some(p), Some(s)) => ltree::index_of(p, s).map(Some),
        _ => Ok(None),
    })
}

// variadic, every argument is a path, a NULL in any argument yields NULL
fn lca_col(args: &[Column]) -> Result<Column> {
    let sig = "lca(ltree, ltree, ...)";
    if args.len() < 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes at least 2 arguments",
            sig
        )));
    }
    let cols: Vec<Vec<Option<&str>>> = args
        .iter()
        .map(|c| text_column(c, sig))
        .collect::<Result<_>>()?;
    let n = row_count(args);
    build_utf8(args, n, |i| {
        let mut paths = Vec::with_capacity(cols.len());
        for col in &cols {
            match col[i] {
                Some(p) => paths.push(p),
                None => return Ok(None),
            }
        }
        ltree::lca(&paths)
    })
}

// variadic, every argument is one label of the resulting path
fn build_path_col(args: &[Column]) -> Result<Column> {
    let sig = "build_path(label, ...)";
    if args.is_empty() {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes at least 1 argument",
            sig
        )));
    }
    let cols: Vec<Vec<Option<&str>>> = args
        .iter()
        .map(|c| text_column(c, sig))
        .collect::<Result<_>>()?;
    let n = row_count(args);
    build_utf8(args, n, |i| {
        let mut labels = Vec::with_capacity(cols.len());
        for col in &cols {
            match col[i] {
                Some(l) => labels.push(l),
                None => return Ok(None),
            }
        }
        ltree::build_path(&labels).map(Some)
    })
}

fn ltree_is_ancestor_col(args: &[Column]) -> Result<Column> {
    let sig = "ltree_is_ancestor(ltree, ltree)";
    arg_check(args, 2, sig)?;
    let ancestors = text_column(&args[0], sig)?;
    let descendants = text_column(&args[1], sig)?;
    let n = row_count(args);
    build_bool(args, n, |i| match (ancestors[i], descendants[i]) {
        (Some(a), Some(d)) => ltree::is_ancestor(a, d).map(Some),
        _ => Ok(None),
    })
}

fn ltree_is_descendant_col(args: &[Column]) -> Result<Column> {
    let sig = "ltree_is_descendant(ltree, ltree)";
    arg_check(args, 2, sig)?;
    let descendants = text_column(&args[0], sig)?;
    let ancestors = text_column(&args[1], sig)?;
    let n = row_count(args);
    build_bool(args, n, |i| match (descendants[i], ancestors[i]) {
        (Some(d), Some(a)) => ltree::is_descendant(d, a).map(Some),
        _ => Ok(None),
    })
}

fn ltree_matches_col(args: &[Column]) -> Result<Column> {
    let sig = "ltree_matches(ltree, lquery)";
    arg_check(args, 2, sig)?;
    let paths = text_column(&args[0], sig)?;
    let queries = text_column(&args[1], sig)?;
    let n = row_count(args);
    build_bool(args, n, |i| match (paths[i], queries[i]) {
        (Some(p), Some(q)) => ltree::matches_lquery(p, q).map(Some),
        _ => Ok(None),
    })
}

fn ltree_matches_any_col(args: &[Column]) -> Result<Column> {
    let sig = "ltree_matches_any(ltree, lquery array)";
    arg_check(args, 2, sig)?;
    let paths = text_column(&args[0], sig)?;
    let query_lists = text_column(&args[1], sig)?;
    let n = row_count(args);
    build_bool(args, n, |i| match (paths[i], query_lists[i]) {
        (Some(p), Some(list_json)) => {
            let list = parse_string_array(list_json).ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "{} expects a JSON string array of lqueries",
                    sig
                ))
            })?;
            let refs: Vec<&str> = list.iter().map(|s| s.as_str()).collect();
            ltree::matches_any_lquery(p, &refs).map(Some)
        }
        _ => Ok(None),
    })
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

// per cell text view, Binary cells hold JSON text, invalid utf8 reads as NULL
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

fn parse_string_array(s: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    v.as_array()?
        .iter()
        .map(|e| e.as_str().map(|x| x.to_string()))
        .collect()
}

fn any_null(args: &[Column], i: usize) -> bool {
    args.iter().any(|c| c.nulls.is_null(i))
}

// each builder applies f per row, a NULL input yields NULL, an Ok(None)
// result yields NULL, an Err propagates and fails the call

fn build_utf8<F: Fn(usize) -> Result<Option<String>>>(
    args: &[Column],
    n: usize,
    f: F,
) -> Result<Column> {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i)? };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Text,
    ))
}

fn build_bool<F: Fn(usize) -> Result<Option<bool>>>(
    args: &[Column],
    n: usize,
    f: F,
) -> Result<Column> {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i)? };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(false);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn build_i64<F: Fn(usize) -> Result<Option<i64>>>(
    args: &[Column],
    n: usize,
    f: F,
) -> Result<Column> {
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let cell = if any_null(args, i) { None } else { f(i)? };
        match cell {
            Some(v) => data.push(v),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Int64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(vals: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(vals.iter().map(|s| s.to_string()).collect()),
            TypeId::Ltree,
        )
    }

    fn int_col(vals: &[i64]) -> Column {
        Column::new(ColumnData::Int64(vals.to_vec()), TypeId::Int64)
    }

    fn utf8_cell(col: &Column, i: usize) -> String {
        match &col.data {
            ColumnData::Utf8(v) => v[i].clone(),
            other => panic!("expected utf8 column, got {:?}", other),
        }
    }

    #[test]
    fn nlevel_counts_and_propagates_null() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(0);
        let col = Column::with_nulls(
            ColumnData::Utf8(vec![String::new(), "a.b.c".to_string()]),
            nulls,
            TypeId::Ltree,
        );
        let out = dispatch("nlevel", &[col], 2).expect("arm").expect("ok");
        assert!(out.nulls.is_null(0));
        assert!(!out.nulls.is_null(1));
        match &out.data {
            ColumnData::Int64(v) => assert_eq!(v[1], 3),
            other => panic!("expected int64 column, got {:?}", other),
        }
    }

    #[test]
    fn subpath_two_and_three_args() {
        let two = dispatch("subpath", &[utf8_col(&["a.b.c.d"]), int_col(&[1])], 1)
            .expect("arm")
            .expect("ok");
        assert_eq!(utf8_cell(&two, 0), "b.c.d");
        let three = dispatch(
            "subpath",
            &[utf8_col(&["a.b.c.d"]), int_col(&[1]), int_col(&[2])],
            1,
        )
        .expect("arm")
        .expect("ok");
        assert_eq!(utf8_cell(&three, 0), "b.c");
    }

    #[test]
    fn subpath_out_of_range_errors() {
        let out = dispatch("subpath", &[utf8_col(&["a.b"]), int_col(&[5])], 1).expect("arm");
        assert!(out.is_err());
    }

    #[test]
    fn ltree_index_reports_position() {
        let out = dispatch(
            "ltree_index",
            &[utf8_col(&["a.b.c.d", "a.b"]), utf8_col(&["b.c", "x"])],
            2,
        )
        .expect("arm")
        .expect("ok");
        match &out.data {
            ColumnData::Int64(v) => assert_eq!(v, &vec![1, -1]),
            other => panic!("expected int64 column, got {:?}", other),
        }
    }

    #[test]
    fn lca_is_variadic() {
        let out = dispatch(
            "lca",
            &[
                utf8_col(&["a.b.c.d"]),
                utf8_col(&["a.b.x"]),
                utf8_col(&["a.b.c"]),
            ],
            1,
        )
        .expect("arm")
        .expect("ok");
        assert_eq!(utf8_cell(&out, 0), "a.b");
    }

    #[test]
    fn lca_disjoint_is_null() {
        let out = dispatch("lca", &[utf8_col(&["a.b"]), utf8_col(&["x.y"])], 1)
            .expect("arm")
            .expect("ok");
        assert!(out.nulls.is_null(0));
    }

    #[test]
    fn build_path_joins_labels() {
        let out = dispatch(
            "build_path",
            &[utf8_col(&["Top"]), utf8_col(&["Science"])],
            1,
        )
        .expect("arm")
        .expect("ok");
        assert_eq!(utf8_cell(&out, 0), "Top.Science");
    }

    #[test]
    fn ancestor_pair_agrees() {
        let anc = dispatch(
            "ltree_is_ancestor",
            &[utf8_col(&["a.b"]), utf8_col(&["a.b.c"])],
            1,
        )
        .expect("arm")
        .expect("ok");
        let desc = dispatch(
            "ltree_is_descendant",
            &[utf8_col(&["a.b.c"]), utf8_col(&["a.b"])],
            1,
        )
        .expect("arm")
        .expect("ok");
        for col in [anc, desc] {
            match &col.data {
                ColumnData::Boolean(v) => assert!(v[0]),
                other => panic!("expected boolean column, got {:?}", other),
            }
        }
    }

    #[test]
    fn matches_lquery_row_wise() {
        let out = dispatch(
            "ltree_matches",
            &[utf8_col(&["a.b.c", "a.b.c"]), utf8_col(&["a.*.c", "x.*"])],
            2,
        )
        .expect("arm")
        .expect("ok");
        match &out.data {
            ColumnData::Boolean(v) => assert_eq!(v, &vec![true, false]),
            other => panic!("expected boolean column, got {:?}", other),
        }
    }

    #[test]
    fn matches_any_reads_json_array() {
        let out = dispatch(
            "ltree_matches_any",
            &[utf8_col(&["a.b.c"]), utf8_col(&[r#"["x.y","a.*"]"#])],
            1,
        )
        .expect("arm")
        .expect("ok");
        match &out.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected boolean column, got {:?}", other),
        }
    }

    #[test]
    fn matches_any_rejects_non_array() {
        let out = dispatch(
            "ltree_matches_any",
            &[utf8_col(&["a.b.c"]), utf8_col(&["not json"])],
            1,
        )
        .expect("arm");
        assert!(out.is_err());
    }

    #[test]
    fn malformed_path_errors() {
        let out = dispatch("nlevel", &[utf8_col(&["a..b"])], 1).expect("arm");
        assert!(out.is_err());
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("not_a_fn", &[], 1).is_none());
    }
}
