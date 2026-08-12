//! In-place conversion between the heap and ZyronLake formats.
//!
//! Conversion is operator initiated and never automatic. It moves rows once,
//! commits the destination, then flips the catalog, so the catalog flag is
//! the single point at which the table changes format. Everything before the
//! flip is invisible to readers and everything after it is authoritative,
//! which is what makes a crash recoverable to exactly one of the two states.
//!
//! The two crash windows and their answers:
//!
//! * Crash before the flip. The lake root exists and the catalog still says
//!   heap, so the root is orphaned work. `reclaim_orphan_root` removes it and
//!   the table is untouched.
//! * Crash after the flip. The table is lake and the heap file is dead
//!   weight, reclaimed by the caller that owns heap files.
//!
//! Nothing here reads or writes a heap file: the caller materializes rows and
//! hands them over, because heap access belongs to the storage layer.

use std::collections::BTreeMap;
use std::fs;

use zyron_common::ZyronError;

use crate::operations::append_rows;
use crate::paths::LakePaths;
use crate::schema::LakeSchema;
use crate::manifest::ClusterSpec;
use crate::transaction_log::{CommitAttempt, OperationKind, TransactionLog};
use crate::writer::ColumnData;

/// Creates a table's lake log and loads it with the rows a heap table held.
///
/// The log is created and populated before the caller flips the catalog, so
/// a crash here leaves an orphaned root that `reclaim_orphan_root` removes.
/// Returns the log so the caller can register it once the flip lands.
pub fn load_lake_from_rows(
    paths: &LakePaths,
    schema: &LakeSchema,
    cluster_spec: Option<&ClusterSpec>,
    properties: &BTreeMap<String, String>,
    table_id: u64,
    rows: &[ColumnData],
    timestamp_us: i64,
) -> Result<TransactionLog, ZyronError> {
    if paths.root().exists() {
        return Err(ZyronError::Internal(format!(
            "cannot convert into an existing lake root at {}",
            paths.root().display()
        )));
    }
    let log = TransactionLog::create(
        paths.clone(),
        CommitAttempt {
            operation: OperationKind::Convert,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us,
            read_predicate: None,
            audit: None,
        },
        schema,
        cluster_spec,
        properties,
    )?;

    let row_count = rows.first().map(|c| c.cells.len()).unwrap_or(0);
    if row_count > 0 {
        append_rows(
            &log,
            CommitAttempt {
                operation: OperationKind::Convert,
                db_txn_id: 0,
                commit_lsn: 0,
                timestamp_us,
                read_predicate: None,
                audit: None,
            },
            table_id,
            rows,
        )?;
    }
    Ok(log)
}

/// Removes a lake root a conversion left behind before its catalog flip.
///
/// Safe precisely because the flip had not happened: no manifest any reader
/// can reach names these files, and the heap the rows came from is still
/// intact. Returns true when a root was removed.
pub fn reclaim_orphan_root(paths: &LakePaths) -> Result<bool, ZyronError> {
    if !paths.root().exists() {
        return Ok(false);
    }
    crate::transaction_log::TransactionLog::remove_shared(paths);
    fs::remove_dir_all(paths.root()).map_err(|e| {
        ZyronError::IoError(format!(
            "cannot reclaim orphaned lake root {}: {}",
            paths.root().display(),
            e
        ))
    })?;
    Ok(true)
}

/// Every live row of a lake table, column major, ready to insert into a heap.
///
/// Reads each file's live rows through the manifest's delete predicates, so
/// a converted table carries exactly what a scan would have returned.
pub fn read_all_rows(
    paths: &LakePaths,
    log: &TransactionLog,
) -> Result<Vec<ColumnData>, ZyronError> {
    let manifest = log.latest_manifest()?;
    let mut columns: Vec<ColumnData> = manifest
        .schema
        .columns
        .iter()
        .map(|c| ColumnData {
            column_id: c.id,
            cells: Vec::new(),
        })
        .collect();

    for entry in &manifest.entries {
        let reader = crate::reader::LakeFileReader::open(paths, entry.partition_id)?;
        let rows = reader.row_count();
        if rows == 0 {
            continue;
        }
        let keep = reader.delete_survivors(&manifest.schema, &manifest, entry)?;
        let mut decoded = Vec::with_capacity(manifest.schema.columns.len());
        for column in &manifest.schema.columns {
            decoded.push(reader.read_column(column)?);
        }
        for row in 0..rows {
            if keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            for (index, column) in decoded.iter().enumerate() {
                columns[index]
                    .cells
                    .push(column.cell(row).map(|cell| cell.to_vec()));
            }
        }
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::LakeColumn;
    use crate::transaction_log::AllCommitted;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![
                LakeColumn {
                    id: 0,
                    name: "id".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "tag".into(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("schema")
    }

    fn rows(ids: &[i64], tags: &[Option<&str>]) -> Vec<ColumnData> {
        vec![
            ColumnData {
                column_id: 0,
                cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
            },
            ColumnData {
                column_id: 1,
                cells: tags
                    .iter()
                    .map(|t| t.map(|s| s.as_bytes().to_vec()))
                    .collect(),
            },
        ]
    }

    #[test]
    fn test_loading_a_lake_root_carries_every_row() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 41);
        let log = load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            41,
            &rows(&[1, 2, 3], &[Some("a"), None, Some("c")]),
            1_000,
        )
        .expect("load");
        assert_eq!(log.latest_version(), 2, "create then load");

        let read = read_all_rows(&paths, &log).expect("read back");
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].cells.len(), 3);
        assert_eq!(read[1].cells[1], None, "the NULL survives the round trip");
        assert_eq!(read[1].cells[0].as_deref(), Some(b"a".as_slice()));
    }

    #[test]
    fn test_an_empty_table_converts_to_a_log_with_no_data_file() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 42);
        let log = load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            42,
            &rows(&[], &[]),
            1_000,
        )
        .expect("load");
        assert_eq!(log.latest_version(), 1, "nothing to append");
        assert!(log.latest_manifest().expect("manifest").entries.is_empty());
        assert!(read_all_rows(&paths, &log).expect("read").iter().all(|c| c.cells.is_empty()));
    }

    #[test]
    fn test_convert_crash_before_catalog_flip_reclaims_the_lake_root() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 43);
        let log = load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            43,
            &rows(&[1, 2], &[Some("a"), Some("b")]),
            1_000,
        )
        .expect("load");
        drop(log);
        assert!(paths.root().exists(), "the conversion wrote its root");

        // The process died before the catalog flip, so the root is work
        // nobody can reach: the catalog still says heap
        assert!(reclaim_orphan_root(&paths).expect("reclaim"));
        assert!(!paths.root().exists());
        // Reclaiming again is a no-op, which is what a restart after a
        // successful reclaim does
        assert!(!reclaim_orphan_root(&paths).expect("reclaim"));
    }

    #[test]
    fn test_converting_into_an_existing_root_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 44);
        load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            44,
            &rows(&[1], &[Some("a")]),
            1_000,
        )
        .expect("load");
        // A second conversion would write into a root that already holds a
        // table's history
        let err = load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            44,
            &rows(&[2], &[Some("b")]),
            2_000,
        )
        .expect_err("an existing root is refused");
        assert!(format!("{err}").contains("existing lake root"));
    }

    #[test]
    fn test_deleted_rows_do_not_travel_back_to_a_heap() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 45);
        let log = load_lake_from_rows(
            &paths,
            &schema(),
            None,
            &BTreeMap::new(),
            45,
            &rows(&[1, 2, 30], &[Some("a"), Some("b"), Some("c")]),
            1_000,
        )
        .expect("load");
        crate::operations::delete_where(
            &log,
            CommitAttempt {
                operation: OperationKind::Delete,
                db_txn_id: 0,
                commit_lsn: 0,
                timestamp_us: 2_000,
                read_predicate: None,
                audit: None,
            },
            &crate::predicate::LakePredicate::Compare {
                column_id: 0,
                op: crate::predicate::CompareOp::Lt,
                value: crate::predicate::LakeValue::Int(10),
            },
            "id < 10",
        )
        .expect("delete");

        let read = read_all_rows(&paths, &log).expect("read");
        assert_eq!(read[0].cells.len(), 1, "only the surviving row travels");
        assert_eq!(read[1].cells[0].as_deref(), Some(b"c".as_slice()));
        let _ = AllCommitted;
    }
}
