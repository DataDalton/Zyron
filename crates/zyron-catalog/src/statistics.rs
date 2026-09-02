//! The ANALYZE statistics file.
//!
//! Statistics are computed by ANALYZE and read by every cardinality
//! estimate. Held only in memory they are lost on restart, which is the
//! difference between an index scan and a sequential one on the first plans
//! after a reboot. One file per table is written so a restart reads back
//! what the last ANALYZE learned.
//!
//! The file carries the universal envelope, so it names its own format and
//! version and both the header and the body are checksummed. A file that
//! does not parse is skipped rather than trusted. Statistics are an
//! optimization, and losing one must not stop a table being read

use std::path::{Path, PathBuf};

use zyron_common::format::envelope;
use zyron_common::format::{FormatKind, FormatVersion};
use zyron_common::{Result, ZyronError};

use crate::ids::TableId;
use crate::stats::{ColumnStats, TableStats};

/// Version statistics files are written at
pub const STATISTICS_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Directory under the data directory that holds statistics files
pub const STATISTICS_DIR: &str = "stats";

/// What one file holds
#[derive(serde::Serialize, serde::Deserialize)]
struct StatisticsBody {
    table: TableStats,
    columns: Vec<ColumnStats>,
}

/// The path one table's statistics live at
pub fn statistics_path(dir: &Path, table_id: TableId) -> PathBuf {
    dir.join(format!("{}.zysts", table_id.0))
}

/// Wraps one table's statistics in a complete statistics file
pub fn encode(table: &TableStats, columns: &[ColumnStats]) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(&StatisticsBody {
        table: table.clone(),
        columns: columns.to_vec(),
    })
    .map_err(|e| ZyronError::Internal(format!("statistics encode failed, {e}")))?;
    Ok(envelope::encode(
        FormatKind::StatisticsFile,
        STATISTICS_FORMAT_VERSION,
        &body,
    ))
}

/// Parses a statistics file, refusing one of another format or version
pub fn decode(bytes: &[u8]) -> Result<(TableStats, Vec<ColumnStats>)> {
    let parsed = envelope::decode_as(bytes, FormatKind::StatisticsFile)
        .map_err(|e| ZyronError::Internal(format!("statistics file, {e}")))?;
    if parsed.header.version != STATISTICS_FORMAT_VERSION {
        return Err(ZyronError::Internal(format!(
            "statistics file is at format version {}, this binary writes and reads {}. \
             Run ANALYZE to rewrite it at the current version",
            parsed.header.version, STATISTICS_FORMAT_VERSION
        )));
    }
    let body: StatisticsBody = serde_json::from_slice(parsed.body)
        .map_err(|e| ZyronError::Internal(format!("statistics decode failed, {e}")))?;
    Ok((body.table, body.columns))
}

/// Writes an already encoded statistics file through a temporary file and a
/// rename, so a crash never leaves a torn file behind.
///
/// Split from `encode` so an async caller runs the serialization on its own
/// thread and hands only the blocking write and fsync to a blocking pool
pub fn write_encoded(dir: &Path, table_id: TableId, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(ZyronError::Io)?;
    let path = statistics_path(dir, table_id);
    let tmp = path.with_extension("zysts.tmp");
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp).map_err(ZyronError::Io)?;
        file.write_all(bytes).map_err(ZyronError::Io)?;
        file.sync_all().map_err(ZyronError::Io)?;
    }
    std::fs::rename(&tmp, &path).map_err(ZyronError::Io)?;
    Ok(())
}

/// Encodes and writes one table's statistics in one call
pub fn write_statistics_file(
    dir: &Path,
    table_id: TableId,
    table: &TableStats,
    columns: &[ColumnStats],
) -> Result<()> {
    let bytes = encode(table, columns)?;
    write_encoded(dir, table_id, &bytes)
}

/// Reads one table's statistics back, or None when the file is absent
pub fn read_statistics_file(
    dir: &Path,
    table_id: TableId,
) -> Result<Option<(TableStats, Vec<ColumnStats>)>> {
    let path = statistics_path(dir, table_id);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ZyronError::Io(e)),
    };
    decode(&bytes).map(Some)
}

/// Removes one table's statistics, which a DROP TABLE does
pub fn remove_statistics_file(dir: &Path, table_id: TableId) {
    let _ = std::fs::remove_file(statistics_path(dir, table_id));
}

/// Reads every statistics file in a directory, table id order.
///
/// A file that does not parse is skipped, so a damaged one costs the plans
/// for its table and nothing else
pub fn read_all(dir: &Path) -> Vec<(TableId, TableStats, Vec<ColumnStats>)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("zysts") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        match decode(&bytes) {
            Ok((table, columns)) => out.push((table.table_id, table, columns)),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "statistics file skipped, run ANALYZE to rebuild it"
                );
            }
        }
    }
    out.sort_by_key(|(id, _, _)| id.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ColumnId;
    use crate::stats::Histogram;

    fn sample() -> (TableStats, Vec<ColumnStats>) {
        (
            TableStats {
                table_id: TableId(7),
                row_count: 1_000,
                page_count: 12,
                avg_row_size: 96,
                last_analyzed: 1_700_000_000,
            },
            vec![
                ColumnStats {
                    table_id: TableId(7),
                    column_id: ColumnId(0),
                    null_fraction: 0.003,
                    distinct_count: 900,
                    avg_width: 8,
                    histogram: Some(Histogram {
                        num_buckets: 2,
                        bounds: vec![b"m".to_vec(), b"z".to_vec()],
                        counts: vec![500, 500],
                    }),
                    most_common_values: vec![b"a".to_vec()],
                    most_common_freqs: vec![0.1],
                },
                ColumnStats {
                    table_id: TableId(7),
                    column_id: ColumnId(1),
                    null_fraction: 0.0,
                    distinct_count: 1,
                    avg_width: 4,
                    histogram: None,
                    most_common_values: vec![],
                    most_common_freqs: vec![],
                },
            ],
        )
    }

    #[test]
    fn test_statistics_round_trip() {
        let (table, columns) = sample();
        let bytes = encode(&table, &columns).expect("encodes");
        let (read_table, read_columns) = decode(&bytes).expect("decodes");
        assert_eq!(read_table.table_id, TableId(7));
        assert_eq!(read_table.row_count, 1_000);
        assert_eq!(read_table.avg_row_size, 96);
        assert_eq!(read_columns.len(), 2);
        assert_eq!(
            read_columns[0]
                .histogram
                .as_ref()
                .expect("histogram")
                .num_buckets,
            2
        );
        assert!(read_columns[1].histogram.is_none());
    }

    #[test]
    fn test_the_file_is_an_envelope() {
        let (table, columns) = sample();
        let bytes = encode(&table, &columns).expect("encodes");
        let (kind, version) = envelope::peek(&bytes).expect("peeks");
        assert_eq!(kind, FormatKind::StatisticsFile);
        assert_eq!(version, STATISTICS_FORMAT_VERSION);
    }

    #[test]
    fn test_corruption_is_refused() {
        let (table, columns) = sample();
        let bytes = encode(&table, &columns).expect("encodes");
        for index in [0usize, 5, 17, bytes.len() - 1] {
            let mut corrupted = bytes.clone();
            corrupted[index] ^= 0x01;
            assert!(decode(&corrupted).is_err(), "byte {index} was not caught");
        }
    }

    #[test]
    fn test_write_read_and_remove_a_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (table, columns) = sample();
        assert!(
            read_statistics_file(dir.path(), TableId(7))
                .expect("absent is not an error")
                .is_none()
        );
        write_statistics_file(dir.path(), TableId(7), &table, &columns).expect("writes");
        let (read_table, read_columns) = read_statistics_file(dir.path(), TableId(7))
            .expect("reads")
            .expect("present");
        assert_eq!(read_table.row_count, 1_000);
        assert_eq!(read_columns.len(), 2);

        let all = read_all(dir.path());
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, TableId(7));

        remove_statistics_file(dir.path(), TableId(7));
        assert!(
            read_statistics_file(dir.path(), TableId(7))
                .expect("absent")
                .is_none()
        );
    }

    #[test]
    fn test_a_corrupt_file_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (table, columns) = sample();
        write_statistics_file(dir.path(), TableId(7), &table, &columns).expect("writes");
        std::fs::write(statistics_path(dir.path(), TableId(9)), b"not a stats file")
            .expect("writes junk");
        let all = read_all(dir.path());
        assert_eq!(all.len(), 1, "the healthy file still loads");
    }
}
