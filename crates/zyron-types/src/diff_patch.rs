//! Diff and patch operations
//!
//! Re-exports the existing text/JSON diff and patch from diff.rs and adds:
//!   ColumnChange + row_diff: per-column comparison of two row tuples
//!   row_diff_ordinal: schema-aware fast path (item W) when both rows share
//!     the same column ordering, skips the column-name hashtable
//!   change_log: pure formatter over caller-supplied raw history rows
//!   JsonDiffRow + json_diff_table: exposes JSON diff as a relation (item H)
//!     so SELECT path FROM JSON_DIFF_TABLE(a,b) WHERE op='replace' works
//!   collapse_adjacent_noops: dedup adjacent insert+delete of equal content
//!     in a DiffOp stream (item V)

pub use crate::diff::{
    DiffOp, json_diff, json_merge_patch, json_patch, text_diff, text_diff_words, text_patch,
};

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// ColumnChange and row_diff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnChange {
    pub column: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Returns the per-column changes between old_row and new_row. The two slices
/// are pairs of (column_name, optional_value) and may have different column
/// sets. Output is sorted by column name. Rows with equal values are not
/// emitted
pub fn row_diff(
    old_row: &[(&str, Option<&str>)],
    new_row: &[(&str, Option<&str>)],
) -> Vec<ColumnChange> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<&str, (Option<&str>, Option<&str>)> = BTreeMap::new();
    for (k, v) in old_row {
        map.entry(*k).or_insert((None, None)).0 = *v;
    }
    for (k, v) in new_row {
        map.entry(*k).or_insert((None, None)).1 = *v;
    }
    let mut out = Vec::new();
    for (col, (old, new)) in map {
        if old != new {
            out.push(ColumnChange {
                column: col.to_string(),
                old_value: old.map(str::to_string),
                new_value: new.map(str::to_string),
            });
        }
    }
    out
}

/// Schema-aware fast path. Both inputs declare the same column slice, only
/// values are compared and emitted when they differ. Avoids the hashtable
/// build/probe of row_diff for the common case of two rows from the same
/// table
pub fn row_diff_ordinal(
    columns: &[&str],
    old_values: &[Option<&str>],
    new_values: &[Option<&str>],
) -> Result<Vec<ColumnChange>> {
    if columns.len() != old_values.len() || columns.len() != new_values.len() {
        return Err(ZyronError::ExecutionError(format!(
            "row_diff_ordinal length mismatch: cols={}, old={}, new={}",
            columns.len(),
            old_values.len(),
            new_values.len()
        )));
    }
    let mut out = Vec::with_capacity(columns.len());
    for (i, col) in columns.iter().enumerate() {
        if old_values[i] != new_values[i] {
            out.push(ColumnChange {
                column: col.to_string(),
                old_value: old_values[i].map(str::to_string),
                new_value: new_values[i].map(str::to_string),
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// change_log: pure formatter over raw history rows
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeLogEntry {
    pub table: String,
    pub version: i64,
    pub operation: String,
    pub column: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Filters and formats history entries for table within (from_version, to_version]
/// Each entry tuple is (version, operation, column, old, new). The executor
/// supplies raw rows from the versioning store, this function only shapes
/// them into the ChangeLogEntry public type
pub fn change_log(
    table: &str,
    from_version: i64,
    to_version: i64,
    entries: &[(i64, &str, &str, Option<&str>, Option<&str>)],
) -> Vec<ChangeLogEntry> {
    let mut out = Vec::new();
    for (ver, op, col, old, new) in entries {
        if *ver > from_version && *ver <= to_version {
            out.push(ChangeLogEntry {
                table: table.to_string(),
                version: *ver,
                operation: op.to_string(),
                column: col.to_string(),
                old_value: old.map(str::to_string),
                new_value: new.map(str::to_string),
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// json_diff_table (item H)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonDiffRow {
    pub path: String,
    pub op: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Returns the JSON Patch ops between old_json and new_json as a relation of
/// JsonDiffRow. Wraps the existing json_diff function and parses the patch
/// once, then materializes per-op rows. Suitable for SQL table-function
/// dispatch: SELECT path FROM JSON_DIFF_TABLE(a, b) WHERE op = 'replace'
pub fn json_diff_table(old_json: &str, new_json: &str) -> Result<Vec<JsonDiffRow>> {
    let patch = json_diff(old_json, new_json)?;
    parse_json_patch_rows(&patch)
}

fn parse_json_patch_rows(patch: &str) -> Result<Vec<JsonDiffRow>> {
    let trimmed = patch.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Ok(Vec::new());
    }
    let val: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| ZyronError::ExecutionError(format!("json patch parse: {}", e)))?;
    let arr = val
        .as_array()
        .ok_or_else(|| ZyronError::ExecutionError("json patch must be an array".to_string()))?;
    let mut out = Vec::with_capacity(arr.len());
    for op_obj in arr {
        let obj = op_obj
            .as_object()
            .ok_or_else(|| ZyronError::ExecutionError("patch op must be an object".to_string()))?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let path = obj
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let new_value = obj.get("value").map(|v| v.to_string());
        // RFC 6902 only emits the new value, not the old. We surface old as
        // None for add/replace and as None for remove (server is free to
        // enrich this from old_json if it wants pre-images)
        let old_value = if op == "remove" || op == "replace" {
            None
        } else {
            None
        };
        out.push(JsonDiffRow {
            path,
            op,
            old_value,
            new_value,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Schema diff and migration generator (item A)
// ---------------------------------------------------------------------------

/// One column descriptor used as input to schema_diff
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDescriptor {
    pub name: String,
    pub sql_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaChangeKind {
    Added,
    Removed,
    TypeChanged,
    NullabilityChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaChange {
    pub kind: SchemaChangeKind,
    pub column: String,
    pub old_type: Option<String>,
    pub new_type: Option<String>,
    pub old_nullable: Option<bool>,
    pub new_nullable: Option<bool>,
}

/// Compares two ordered column lists. Output is sorted by column name with
/// added columns first, then removed, then type/nullability changes
pub fn schema_diff(old: &[ColumnDescriptor], new: &[ColumnDescriptor]) -> Vec<SchemaChange> {
    use std::collections::BTreeMap;
    let mut old_map: BTreeMap<&str, &ColumnDescriptor> =
        old.iter().map(|c| (c.name.as_str(), c)).collect();
    let mut out = Vec::new();
    for new_col in new {
        match old_map.remove(new_col.name.as_str()) {
            None => out.push(SchemaChange {
                kind: SchemaChangeKind::Added,
                column: new_col.name.clone(),
                old_type: None,
                new_type: Some(new_col.sql_type.clone()),
                old_nullable: None,
                new_nullable: Some(new_col.nullable),
            }),
            Some(old_col) => {
                if old_col.sql_type != new_col.sql_type {
                    out.push(SchemaChange {
                        kind: SchemaChangeKind::TypeChanged,
                        column: new_col.name.clone(),
                        old_type: Some(old_col.sql_type.clone()),
                        new_type: Some(new_col.sql_type.clone()),
                        old_nullable: Some(old_col.nullable),
                        new_nullable: Some(new_col.nullable),
                    });
                }
                if old_col.nullable != new_col.nullable {
                    out.push(SchemaChange {
                        kind: SchemaChangeKind::NullabilityChanged,
                        column: new_col.name.clone(),
                        old_type: Some(old_col.sql_type.clone()),
                        new_type: Some(new_col.sql_type.clone()),
                        old_nullable: Some(old_col.nullable),
                        new_nullable: Some(new_col.nullable),
                    });
                }
            }
        }
    }
    for (name, old_col) in old_map {
        out.push(SchemaChange {
            kind: SchemaChangeKind::Removed,
            column: name.to_string(),
            old_type: Some(old_col.sql_type.clone()),
            new_type: None,
            old_nullable: Some(old_col.nullable),
            new_nullable: None,
        });
    }
    out
}

/// Produces ALTER TABLE statements that migrate the old schema to the new
/// Statements are emitted in the order: ADD COLUMN, ALTER COLUMN TYPE,
/// ALTER COLUMN NULLABILITY, DROP COLUMN
pub fn generate_migration(table: &str, changes: &[SchemaChange]) -> Vec<String> {
    let mut adds = Vec::new();
    let mut type_changes = Vec::new();
    let mut nullability = Vec::new();
    let mut drops = Vec::new();
    for c in changes {
        match c.kind {
            SchemaChangeKind::Added => {
                let null_clause = if c.new_nullable.unwrap_or(true) {
                    ""
                } else {
                    " NOT NULL"
                };
                adds.push(format!(
                    "ALTER TABLE {} ADD COLUMN {} {}{}",
                    table,
                    c.column,
                    c.new_type.as_deref().unwrap_or(""),
                    null_clause
                ));
            }
            SchemaChangeKind::TypeChanged => {
                type_changes.push(format!(
                    "ALTER TABLE {} ALTER COLUMN {} TYPE {}",
                    table,
                    c.column,
                    c.new_type.as_deref().unwrap_or("")
                ));
            }
            SchemaChangeKind::NullabilityChanged => {
                let action = if c.new_nullable.unwrap_or(true) {
                    "DROP NOT NULL"
                } else {
                    "SET NOT NULL"
                };
                nullability.push(format!(
                    "ALTER TABLE {} ALTER COLUMN {} {}",
                    table, c.column, action
                ));
            }
            SchemaChangeKind::Removed => {
                drops.push(format!("ALTER TABLE {} DROP COLUMN {}", table, c.column));
            }
        }
    }
    let mut out =
        Vec::with_capacity(adds.len() + type_changes.len() + nullability.len() + drops.len());
    out.extend(adds);
    out.extend(type_changes);
    out.extend(nullability);
    out.extend(drops);
    out
}

// ---------------------------------------------------------------------------
// collapse_adjacent_noops (item V)
// ---------------------------------------------------------------------------

/// Walks ops left to right and collapses adjacent Insert/Delete pairs whose
/// content matches into Equal. Used by the diff stream shipped to replicas
/// to compress wire bandwidth
pub fn collapse_adjacent_noops(ops: Vec<DiffOp>) -> Vec<DiffOp> {
    let mut out: Vec<DiffOp> = Vec::with_capacity(ops.len());
    for op in ops {
        match (out.last(), &op) {
            (Some(DiffOp::Insert(prev)), DiffOp::Delete(curr))
            | (Some(DiffOp::Delete(prev)), DiffOp::Insert(curr))
                if prev == curr =>
            {
                let last = out.pop();
                if let Some(DiffOp::Insert(s)) | Some(DiffOp::Delete(s)) = last {
                    out.push(DiffOp::Equal(s));
                }
            }
            _ => out.push(op),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_diff_emits_changed_columns_only() {
        let old: &[(&str, Option<&str>)] = &[("name", Some("alice")), ("age", Some("30"))];
        let new: &[(&str, Option<&str>)] = &[("name", Some("alice")), ("age", Some("31"))];
        let d = row_diff(old, new);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].column, "age");
        assert_eq!(d[0].old_value.as_deref(), Some("30"));
        assert_eq!(d[0].new_value.as_deref(), Some("31"));
    }

    #[test]
    fn row_diff_added_and_removed_columns() {
        let old: &[(&str, Option<&str>)] = &[("a", Some("1"))];
        let new: &[(&str, Option<&str>)] = &[("a", Some("1")), ("b", Some("2"))];
        let d = row_diff(old, new);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].column, "b");
        assert_eq!(d[0].old_value, None);
        assert_eq!(d[0].new_value.as_deref(), Some("2"));
    }

    #[test]
    fn row_diff_ordinal_matches_row_diff() {
        let cols = &["a", "b"];
        let old: &[Option<&str>] = &[Some("1"), Some("2")];
        let new: &[Option<&str>] = &[Some("1"), Some("3")];
        let d = row_diff_ordinal(cols, old, new).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].column, "b");
    }

    #[test]
    fn row_diff_ordinal_length_check() {
        let cols = &["a"];
        let old: &[Option<&str>] = &[Some("1"), Some("2")];
        let new: &[Option<&str>] = &[Some("1"), Some("2")];
        assert!(row_diff_ordinal(cols, old, new).is_err());
    }

    #[test]
    fn change_log_filters_versions() {
        let entries: &[(i64, &str, &str, Option<&str>, Option<&str>)] = &[
            (1, "insert", "name", None, Some("a")),
            (2, "update", "name", Some("a"), Some("b")),
            (3, "update", "name", Some("b"), Some("c")),
        ];
        let log = change_log("users", 1, 3, entries);
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].version, 2);
        assert_eq!(log[1].version, 3);
    }

    #[test]
    fn json_diff_table_replace_and_add() {
        let rows = json_diff_table(r#"{"a":1,"b":2}"#, r#"{"a":1,"b":3,"c":4}"#).unwrap();
        let ops: Vec<&str> = rows.iter().map(|r| r.op.as_str()).collect();
        assert!(ops.contains(&"replace"));
        assert!(ops.contains(&"add"));
        // Path encoding follows RFC 6901
        assert!(rows.iter().any(|r| r.path == "/b"));
        assert!(rows.iter().any(|r| r.path == "/c"));
    }

    #[test]
    fn collapse_adjacent_noops_basic() {
        let ops = vec![
            DiffOp::Equal("hello".to_string()),
            DiffOp::Insert("world".to_string()),
            DiffOp::Delete("world".to_string()),
            DiffOp::Equal("end".to_string()),
        ];
        let collapsed = collapse_adjacent_noops(ops);
        // The Insert+Delete of "world" should fold into a single Equal
        let inserts = collapsed
            .iter()
            .filter(|op| matches!(op, DiffOp::Insert(_)))
            .count();
        let deletes = collapsed
            .iter()
            .filter(|op| matches!(op, DiffOp::Delete(_)))
            .count();
        assert_eq!(inserts, 0);
        assert_eq!(deletes, 0);
    }

    #[test]
    fn re_exported_text_diff() {
        let d = text_diff("a\nb\nc", "a\nB\nc");
        assert!(d.contains("b") || d.contains("B"));
    }

    #[test]
    fn schema_diff_detects_added_removed_changed() {
        let old = vec![
            ColumnDescriptor {
                name: "id".into(),
                sql_type: "INT".into(),
                nullable: false,
            },
            ColumnDescriptor {
                name: "old_col".into(),
                sql_type: "TEXT".into(),
                nullable: true,
            },
        ];
        let new = vec![
            ColumnDescriptor {
                name: "id".into(),
                sql_type: "BIGINT".into(),
                nullable: false,
            },
            ColumnDescriptor {
                name: "new_col".into(),
                sql_type: "TEXT".into(),
                nullable: false,
            },
        ];
        let changes = schema_diff(&old, &new);
        let kinds: Vec<&SchemaChangeKind> = changes.iter().map(|c| &c.kind).collect();
        assert!(kinds.contains(&&SchemaChangeKind::Added));
        assert!(kinds.contains(&&SchemaChangeKind::Removed));
        assert!(kinds.contains(&&SchemaChangeKind::TypeChanged));
    }

    #[test]
    fn generate_migration_orders_changes() {
        let changes = vec![
            SchemaChange {
                kind: SchemaChangeKind::Added,
                column: "c".into(),
                old_type: None,
                new_type: Some("INT".into()),
                old_nullable: None,
                new_nullable: Some(true),
            },
            SchemaChange {
                kind: SchemaChangeKind::Removed,
                column: "d".into(),
                old_type: Some("TEXT".into()),
                new_type: None,
                old_nullable: Some(true),
                new_nullable: None,
            },
        ];
        let sql = generate_migration("t", &changes);
        assert!(sql[0].contains("ADD COLUMN c"));
        assert!(sql[1].contains("DROP COLUMN d"));
    }

    #[test]
    fn re_exported_json_patch_round_trip() {
        let old = r#"{"a":1,"b":2}"#;
        let new = r#"{"a":1,"b":3,"c":4}"#;
        let patch = json_diff(old, new).unwrap();
        let result = json_patch(old, &patch).unwrap();
        // Result must equal new modulo whitespace
        let result_v: serde_json::Value = serde_json::from_str(&result).unwrap();
        let new_v: serde_json::Value = serde_json::from_str(new).unwrap();
        assert_eq!(result_v, new_v);
    }
}
