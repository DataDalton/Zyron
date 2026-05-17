//! Shared DRY RUN / preview result used by ARCHIVE/MOVE/retention/erasure and
//! the dashboard estimators. Pure data; the caller fills it from the same
//! predicate/scan path the real op uses, then skips the mutation.

#[derive(Debug, Clone, Default)]
pub struct PreviewResult {
    pub rows: u64,
    pub bytes: u64,
    pub segments_affected: u64,
    /// A small sample of affected primary-key strings for human inspection.
    pub sample_keys: Vec<String>,
}

impl PreviewResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_row(&mut self, bytes: u64, key: Option<String>) {
        self.rows += 1;
        self.bytes += bytes;
        if let Some(k) = key {
            if self.sample_keys.len() < 16 {
                self.sample_keys.push(k);
            }
        }
    }

    /// One-line human summary for the client.
    pub fn summary(&self) -> String {
        format!(
            "DRY RUN: {} rows, {} bytes, {} segments",
            self.rows, self.bytes, self.segments_affected
        )
    }
}
