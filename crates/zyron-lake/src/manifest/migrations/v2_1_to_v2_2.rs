//! Moves a manifest checkpoint from 2.1 to 2.2.
//!
//! A 2.2 manifest records the schema id each file was written under and
//! the type each column had through each schema id it changed at, in a
//! section a 2.1 reader has no footer offset for. No column type ever
//! changed under the 2.1 layout, so every file a 2.1 checkpoint lists holds
//! the shape its schema declares. Moving it forward is a decode at 2.1,
//! which stamps every file with the checkpoint's schema id and an empty
//! history, and an encode at 2.2

use zyron_common::format::FormatKind;
use zyron_common::format::envelope;
use zyron_common::format::registry::{FormatFixture, FormatMigrator};

use crate::format::{LAKE_MANIFEST_FORMAT_VERSION_2_1, LAKE_MANIFEST_FORMAT_VERSION_2_2};
use crate::manifest::ManifestFile;

/// A checkpoint the 0.18.0 writer produced, so the 2.1 reader and this
/// step are exercised against bytes that writer laid down rather than
/// bytes the current writer round-tripped
static FIXTURE_2_1: &[u8] = include_bytes!("../fixtures/v2_1.bin");

inventory::submit! {
    FormatFixture {
        kind: FormatKind::LakeManifest,
        version: LAKE_MANIFEST_FORMAT_VERSION_2_1,
        bytes: FIXTURE_2_1,
        path: "crates/zyron-lake/src/manifest/fixtures/v2_1.bin",
    }
}

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::LakeManifest,
        from: LAKE_MANIFEST_FORMAT_VERSION_2_1,
        to: LAKE_MANIFEST_FORMAT_VERSION_2_2,
        reversible: false,
        forward: manifest_2_1_to_2_2,
        backward: None,
        no_body_change: false,
        description: "file entries gained the schema id they were written under and the manifest a column type history",
    }
}

/// Re-encodes a 2.1 checkpoint at 2.2, the whole file in and the whole
/// file out. The manifest's own checksum and trailer are rebuilt by the
/// encoder, so a damaged checkpoint is refused by the decode rather than
/// carried forward
pub fn manifest_2_1_to_2_2(file: &[u8]) -> Result<Vec<u8>, String> {
    let (header, _) = envelope::decode_header(file).map_err(|e| e.to_string())?;
    if header.kind != FormatKind::LakeManifest {
        return Err(format!("a {} file, not a lake manifest", header.kind));
    }
    if header.version != LAKE_MANIFEST_FORMAT_VERSION_2_1 {
        return Err(format!(
            "at version {}, this step moves {} forward",
            header.version, LAKE_MANIFEST_FORMAT_VERSION_2_1
        ));
    }
    let manifest = ManifestFile::decode(file, "manifest migration").map_err(|e| e.to_string())?;
    Ok(manifest.encode_at(LAKE_MANIFEST_FORMAT_VERSION_2_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_is_a_2_1_checkpoint_the_reader_opens() {
        let (kind, version) = envelope::peek(FIXTURE_2_1).expect("peeks");
        assert_eq!(kind, FormatKind::LakeManifest);
        assert_eq!(version, LAKE_MANIFEST_FORMAT_VERSION_2_1);
        let manifest = ManifestFile::decode(FIXTURE_2_1, "fixture").expect("decodes");
        assert_eq!(manifest.entries.len(), 2);
        assert_eq!(
            manifest.entries.iter().map(|e| e.row_count).sum::<u64>(),
            200
        );
        assert_eq!(manifest.delete_predicates.len(), 1);
        assert!(
            manifest
                .entries
                .iter()
                .all(|e| e.schema_id == manifest.schema.schema_id),
            "every file of a 2.1 checkpoint holds the shape its schema declares"
        );
        assert!(manifest.type_history.is_empty());
        assert!(
            manifest.entries[0]
                .column_stats
                .iter()
                .any(|s| s.sum.is_some()),
            "a 2.1 checkpoint records sums"
        );
    }

    #[test]
    fn moving_forward_keeps_every_entry_and_stamps_2_2() {
        let moved = manifest_2_1_to_2_2(FIXTURE_2_1).expect("moves");
        let (_, version) = envelope::peek(&moved).expect("peeks");
        assert_eq!(version, LAKE_MANIFEST_FORMAT_VERSION_2_2);
        let before = ManifestFile::decode(FIXTURE_2_1, "fixture").expect("decodes");
        let after = ManifestFile::decode(&moved, "moved").expect("decodes");
        assert_eq!(after, before);
    }

    #[test]
    fn a_checkpoint_already_at_2_2_is_refused_by_the_step() {
        let moved = manifest_2_1_to_2_2(FIXTURE_2_1).expect("moves");
        let refused = manifest_2_1_to_2_2(&moved).expect_err("a 2.2 checkpoint is not 2.1");
        assert!(refused.contains("2.2"), "{refused}");
    }

    #[test]
    fn a_damaged_checkpoint_is_refused_rather_than_carried() {
        let mut damaged = FIXTURE_2_1.to_vec();
        let middle = damaged.len() / 2;
        damaged[middle] ^= 0x5A;
        assert!(manifest_2_1_to_2_2(&damaged).is_err());
    }
}
