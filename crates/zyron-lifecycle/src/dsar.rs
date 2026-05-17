//! GDPR DSAR export (Article 15/20): collect all of a subject's data into a
//! portable bundle. Read-side pair of erasure::forget_user. Row collection is
//! delegated to an injected collector; the bundle is written via the archive
//! object layer.

use zyron_common::Result;

use crate::archive::{ArchiveResult, archive_rows};
use crate::erasure::ErasureTarget;

#[derive(Debug, Clone, Default)]
pub struct DsarResult {
    pub tables_exported: u64,
    pub rows_exported: u64,
    pub bytes_written: u64,
}

/// Collects a subject's rows for one target as serialized records. Implemented
/// by the dispatch/executor bridge.
pub trait DsarCollector {
    fn collect_rows(&self, target: &ErasureTarget) -> Result<Vec<Vec<u8>>>;
}

/// Exports every target's rows for the subject and writes one bundle object
/// per table under `destination_uri`. Returns aggregate counts.
pub async fn export_user(
    subject: &str,
    targets: &[ErasureTarget],
    destination_uri: &str,
    collector: &dyn DsarCollector,
) -> Result<DsarResult> {
    let mut result = DsarResult::default();
    for t in targets {
        let rows = collector.collect_rows(t)?;
        if rows.is_empty() {
            continue;
        }
        let dest = format!(
            "{}/{}/{}.zylog",
            destination_uri.trim_end_matches('/'),
            sanitize(subject),
            sanitize(&t.table_name)
        );
        let ArchiveResult {
            rows_archived,
            bytes_written,
            ..
        } = archive_rows(&dest, &rows).await?;
        result.tables_exported += 1;
        result.rows_exported += rows_archived;
        result.bytes_written += bytes_written;
    }
    Ok(result)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
