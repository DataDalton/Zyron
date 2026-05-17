//! Archive export/import over real object storage (S3/GCS/Azure/FS) using the
//! centralized opendal layer in zyron-streaming. Rows are serialized as
//! newline-delimited records; the caller chooses the field encoding.

use std::collections::HashMap;

use zyron_common::{Result, ZyronError};
use zyron_streaming::external_source::build_object_operator;

/// Result of an archive export.
#[derive(Debug, Clone, Default)]
pub struct ArchiveResult {
    pub rows_archived: u64,
    pub bytes_written: u64,
    pub files_created: u64,
}

/// Result of an archive restore.
#[derive(Debug, Clone, Default)]
pub struct RestoreResult {
    pub rows_restored: u64,
    pub bytes_read: u64,
}

/// Splits a destination URI into (operator-base, object-key). The operator is
/// rooted at the bucket/container; the key is the trailing path.
fn split_destination(uri: &str) -> (String, String) {
    // For object stores opendal needs the bucket as the root and the rest as
    // the key. build_object_operator already returns the key prefix.
    (uri.to_string(), String::new())
}

/// Writes serialized rows to `uri`. Each element of `rows` is one record's
/// already-encoded bytes; records are joined with newline.
pub async fn archive_rows(uri: &str, rows: &[Vec<u8>]) -> Result<ArchiveResult> {
    let (base, _k) = split_destination(uri);
    let (op, prefix) = build_object_operator(&base, &[], &HashMap::new())?;
    let key = if prefix.is_empty() {
        "archive.zylog".to_string()
    } else {
        prefix
    };
    let mut payload = Vec::with_capacity(rows.iter().map(|r| r.len() + 1).sum());
    for r in rows {
        payload.extend_from_slice(r);
        payload.push(b'\n');
    }
    let bytes_written = payload.len() as u64;
    op.write(&key, payload)
        .await
        .map_err(|e| ZyronError::Internal(format!("archive write failed: {e}")))?;
    Ok(ArchiveResult {
        rows_archived: rows.len() as u64,
        bytes_written,
        files_created: 1,
    })
}

/// Reads an archived object back into per-record byte vectors.
pub async fn restore_from(uri: &str) -> Result<(Vec<Vec<u8>>, RestoreResult)> {
    let (base, _k) = split_destination(uri);
    let (op, prefix) = build_object_operator(&base, &[], &HashMap::new())?;
    let key = if prefix.is_empty() {
        "archive.zylog".to_string()
    } else {
        prefix
    };
    let buf = op
        .read(&key)
        .await
        .map_err(|e| ZyronError::Internal(format!("archive read failed: {e}")))?;
    let bytes = buf.to_vec();
    let bytes_read = bytes.len() as u64;
    let mut rows = Vec::new();
    for line in bytes.split(|b| *b == b'\n') {
        if !line.is_empty() {
            rows.push(line.to_vec());
        }
    }
    let rows_restored = rows.len() as u64;
    Ok((
        rows,
        RestoreResult {
            rows_restored,
            bytes_read,
        },
    ))
}

/// Deletes an archived object (used when an archive's retention elapses).
pub async fn delete_archive(uri: &str) -> Result<()> {
    let (base, _k) = split_destination(uri);
    let (op, prefix) = build_object_operator(&base, &[], &HashMap::new())?;
    let key = if prefix.is_empty() {
        "archive.zylog".to_string()
    } else {
        prefix
    };
    op.delete(&key)
        .await
        .map_err(|e| ZyronError::Internal(format!("archive delete failed: {e}")))?;
    Ok(())
}
