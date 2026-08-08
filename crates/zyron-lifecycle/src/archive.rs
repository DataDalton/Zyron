//! Archive export/import over real object storage (S3/GCS/Azure/FS) using the
//! centralized opendal layer in zyron-streaming. Rows are serialized as
//! newline-delimited records; the caller chooses the field encoding.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::{Result, ZyronError};
use zyron_streaming::external_source::build_object_operator;

/// Monotonic counter that disambiguates archive object keys written within the
/// same nanosecond, so concurrent or rapid archives to a prefix-less
/// destination never collide.
static ARCHIVE_KEY_SEQ: AtomicU64 = AtomicU64::new(0);

/// Builds the object key for an archive write. When the destination carries an
/// explicit object name the prefix is used verbatim so a restore from the same
/// URI reads it back. When the destination is a bare directory or bucket the
/// key is made unique with a timestamp and a monotonic counter, matching how
/// the CDC S3 sinks key their objects, so distinct archives never overwrite.
fn archive_object_key(prefix: &str) -> String {
    if !prefix.is_empty() {
        return prefix.to_string();
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = ARCHIVE_KEY_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("archive-{ts:039}-{seq:010}.zylog")
}

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
    let key = archive_object_key(&prefix);
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

/// Reads archived objects back into per-record byte vectors. When the source
/// carries an explicit object name that single object is read. When the source
/// is a bare directory or bucket every archive object written there is listed
/// and concatenated, so the unique-keyed writes from archive_rows round-trip.
pub async fn restore_from(uri: &str) -> Result<(Vec<Vec<u8>>, RestoreResult)> {
    let (base, _k) = split_destination(uri);
    let (op, prefix) = build_object_operator(&base, &[], &HashMap::new())?;

    let keys: Vec<String> = if prefix.is_empty() {
        let entries = op
            .list("")
            .await
            .map_err(|e| ZyronError::Internal(format!("archive list failed: {e}")))?;
        let mut keys: Vec<String> = entries
            .into_iter()
            .map(|e| e.path().to_string())
            .filter(|k| k.ends_with(".zylog") && !k.ends_with('/'))
            .collect();
        keys.sort();
        keys
    } else {
        vec![prefix]
    };

    let mut rows = Vec::new();
    let mut bytes_read: u64 = 0;
    for key in &keys {
        let buf = op
            .read(key)
            .await
            .map_err(|e| ZyronError::Internal(format!("archive read failed: {e}")))?;
        let bytes = buf.to_vec();
        bytes_read += bytes.len() as u64;
        for line in bytes.split(|b| *b == b'\n') {
            if !line.is_empty() {
                rows.push(line.to_vec());
            }
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

/// Deletes archived objects (used when an archive's retention elapses). An
/// explicit object name deletes that single object. A bare directory or bucket
/// deletes every archive object written there, matching the unique keys
/// archive_rows produces for prefix-less destinations.
pub async fn delete_archive(uri: &str) -> Result<()> {
    let (base, _k) = split_destination(uri);
    let (op, prefix) = build_object_operator(&base, &[], &HashMap::new())?;

    if prefix.is_empty() {
        let entries = op
            .list("")
            .await
            .map_err(|e| ZyronError::Internal(format!("archive list failed: {e}")))?;
        for entry in entries {
            let key = entry.path().to_string();
            if key.ends_with(".zylog") && !key.ends_with('/') {
                op.delete(&key)
                    .await
                    .map_err(|e| ZyronError::Internal(format!("archive delete failed: {e}")))?;
            }
        }
        return Ok(());
    }

    op.delete(&prefix)
        .await
        .map_err(|e| ZyronError::Internal(format!("archive delete failed: {e}")))?;
    Ok(())
}
