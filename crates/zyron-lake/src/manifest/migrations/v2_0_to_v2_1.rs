//! Moves a manifest checkpoint from 2.0 to 2.1.
//!
//! A 2.1 manifest records an exact sum beside each column's bounds, behind
//! a presence flag a 2.0 reader refuses. A 2.0 manifest holds every section
//! a 2.1 reader expects and no sum, so moving it forward is a decode at 2.0
//! and an encode at 2.1, with every sum absent until the next commit
//! rewrites the table's statistics

use zyron_common::format::FormatKind;
use zyron_common::format::envelope;
use zyron_common::format::registry::{FormatFixture, FormatMigrator};

use crate::format::{LAKE_MANIFEST_FORMAT_VERSION_2_0, LAKE_MANIFEST_FORMAT_VERSION_2_1};
use crate::manifest::ManifestFile;

/// A checkpoint the 0.11.0 writer produced, so the 2.0 reader and this
/// step are exercised against bytes that writer laid down rather than
/// bytes the current writer round-tripped
static FIXTURE_2_0: &[u8] = include_bytes!("../fixtures/v2_0.bin");

inventory::submit! {
    FormatFixture {
        kind: FormatKind::LakeManifest,
        version: LAKE_MANIFEST_FORMAT_VERSION_2_0,
        bytes: FIXTURE_2_0,
        path: "crates/zyron-lake/src/manifest/fixtures/v2_0.bin",
    }
}

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::LakeManifest,
        from: LAKE_MANIFEST_FORMAT_VERSION_2_0,
        to: LAKE_MANIFEST_FORMAT_VERSION_2_1,
        reversible: false,
        forward: manifest_2_0_to_2_1,
        backward: None,
        no_body_change: false,
        description: "column statistics gained an exact sum behind a presence flag",
    }
}

/// Re-encodes a 2.0 checkpoint at 2.1, the whole file in and the whole
/// file out. The manifest's own checksum and trailer are rebuilt by the
/// encoder, so a damaged checkpoint is refused by the decode rather than
/// carried forward
pub fn manifest_2_0_to_2_1(file: &[u8]) -> Result<Vec<u8>, String> {
    let (header, _) = envelope::decode_header(file).map_err(|e| e.to_string())?;
    if header.kind != FormatKind::LakeManifest {
        return Err(format!("a {} file, not a lake manifest", header.kind));
    }
    if header.version != LAKE_MANIFEST_FORMAT_VERSION_2_0 {
        return Err(format!(
            "at version {}, this step moves {} forward",
            header.version, LAKE_MANIFEST_FORMAT_VERSION_2_0
        ));
    }
    let manifest = ManifestFile::decode(file, "manifest migration").map_err(|e| e.to_string())?;
    Ok(manifest.encode_at(LAKE_MANIFEST_FORMAT_VERSION_2_1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_is_a_2_0_checkpoint_the_reader_opens() {
        let (kind, version) = envelope::peek(FIXTURE_2_0).expect("peeks");
        assert_eq!(kind, FormatKind::LakeManifest);
        assert_eq!(version, LAKE_MANIFEST_FORMAT_VERSION_2_0);
        let manifest = ManifestFile::decode(FIXTURE_2_0, "fixture").expect("decodes");
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.entries[0].row_count, 200);
        assert!(
            manifest.entries[0]
                .column_stats
                .iter()
                .all(|s| s.sum.is_none()),
            "a 2.0 checkpoint records no sum"
        );
    }

    #[test]
    fn moving_forward_keeps_every_entry_and_stamps_2_1() {
        let moved = manifest_2_0_to_2_1(FIXTURE_2_0).expect("moves");
        let (_, version) = envelope::peek(&moved).expect("peeks");
        assert_eq!(version, LAKE_MANIFEST_FORMAT_VERSION_2_1);
        let before = ManifestFile::decode(FIXTURE_2_0, "fixture").expect("decodes");
        let after = ManifestFile::decode(&moved, "moved").expect("decodes");
        assert_eq!(after.snapshot_id, before.snapshot_id);
        assert_eq!(after.schema.schema_id, before.schema.schema_id);
        assert_eq!(after.entries, before.entries);
        assert_eq!(after.cluster_spec, before.cluster_spec);
    }

    #[test]
    fn a_checkpoint_already_at_2_1_is_refused_by_the_step() {
        let moved = manifest_2_0_to_2_1(FIXTURE_2_0).expect("moves");
        let refused = manifest_2_0_to_2_1(&moved).expect_err("a 2.1 checkpoint is not 2.0");
        assert!(refused.contains("2.1"), "{refused}");
    }

    #[test]
    fn a_damaged_checkpoint_is_refused_rather_than_carried() {
        let mut damaged = FIXTURE_2_0.to_vec();
        let middle = damaged.len() / 2;
        damaged[middle] ^= 0x5A;
        assert!(manifest_2_0_to_2_1(&damaged).is_err());
    }
}
