//! CDC format registrations.
//!
//! A snapshot manifest is a point-in-time capture that a consumer replays
//! from, so old and new versions coexist and a manifest is only moved
//! forward when it is read. The stream checkpoint is rewritten on every
//! progress update, so it migrates eagerly the next time a stream restarts.
//!
//! A change feed segment is append-only until it is sealed, so it carries an
//! envelope header and a trailer of its own rather than a body checksum the
//! envelope owns. The feed manifest beside it is rewritten whenever a segment
//! seals, which makes it eager

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};

const GATE: &str = "0.11.0";

/// The Zyron version the change feed became a segmented log with a manifest
const FEED_GATE: &str = "0.18.0";

/// Version the snapshot manifest is written at
pub const CDC_SNAPSHOT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version the outbound stream state is written at. A version 1.1 stream
/// names the change stream it consumes, which is where its position lives
pub const CDC_CHECKPOINT_FORMAT_VERSION: FormatVersion = FormatVersion::new(1, 1);

/// The version that recorded a replication slot per outbound stream
pub const CDC_CHECKPOINT_FORMAT_VERSION_1: FormatVersion = FormatVersion::V1;

/// The Zyron version an outbound stream started consuming a named change
/// stream rather than a slot of its own
const STREAM_STATE_GATE: &str = "0.18.0";

/// The day the 1.0 stream state reader and its migration leave the tree.
/// The state is rewritten on every definition change and moved forward on
/// the first read, so every node is past it long before
const STREAM_STATE_1_0_RETIREMENT: &str = "2027-03-01";

/// Version a change feed segment is written at
pub const CHANGE_FEED_SEGMENT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version a change feed manifest is written at
pub const CHANGE_FEED_MANIFEST_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version a derived feed's record index is written at
pub const CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::Snapshot,
        writer_current_version: CDC_SNAPSHOT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CDC_SNAPSHOT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "point-in-time capture of a table set, historical, migrated on read",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::StreamingCdcCheckpoint,
        writer_current_version: CDC_CHECKPOINT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::new(
            CDC_CHECKPOINT_FORMAT_VERSION_1,
            CDC_CHECKPOINT_FORMAT_VERSION,
        ),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: false,
        binary_version_gate: STREAM_STATE_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: Some(STREAM_STATE_1_0_RETIREMENT),
        downgrade_write_supported: false,
        notes: "outbound stream definitions naming the change stream each consumes,                 rewritten on every definition change",
    }
}

/// A stream state file the 0.17.0 writer laid down, so the version 1 reader
/// and the step forward are exercised against bytes that writer produced
pub(crate) static STREAM_STATE_FIXTURE_1: &[u8] = include_bytes!("fixtures/cdc_streams_v1.bin");

inventory::submit! {
    zyron_common::format::registry::FormatFixture {
        kind: FormatKind::StreamingCdcCheckpoint,
        version: CDC_CHECKPOINT_FORMAT_VERSION_1,
        bytes: STREAM_STATE_FIXTURE_1,
        path: "crates/zyron-cdc/src/fixtures/cdc_streams_v1.bin",
    }
}

inventory::submit! {
    zyron_common::format::registry::FormatMigrator {
        kind: FormatKind::StreamingCdcCheckpoint,
        from: CDC_CHECKPOINT_FORMAT_VERSION_1,
        to: CDC_CHECKPOINT_FORMAT_VERSION,
        reversible: false,
        forward: cdc_streams_1_0_to_1_1,
        backward: None,
        no_body_change: false,
        description: "an outbound stream names the change stream it consumes in place of a                       replication slot of its own",
    }
}

/// Moves a version 1.0 stream state body forward.
///
/// A version 1.0 stream recorded a replication slot as its position. At
/// version 1.1 its position is the change stream named after it, which the
/// stream pump creates on the first pass that finds no such stream, at the
/// place the slot had reached, and the slot is retired once it has
pub fn cdc_streams_1_0_to_1_1(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut list: Vec<serde_json::Value> =
        serde_json::from_slice(body).map_err(|e| format!("stream state, {e}"))?;
    for stream in list.iter_mut() {
        let Some(object) = stream.as_object_mut() else {
            return Err("stream state holds an entry that is not an object".to_string());
        };
        let name = object
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| "stream state holds an entry with no name".to_string())?
            .to_string();
        object.remove("slot_name");
        object.insert(
            "change_stream".to_string(),
            serde_json::Value::String(crate::cdc_stream::implicit_change_stream_name(&name)),
        );
    }
    serde_json::to_vec(&list).map_err(|e| format!("stream state, {e}"))
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ChangeFeedSegment,
        writer_current_version: CHANGE_FEED_SEGMENT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CHANGE_FEED_SEGMENT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: FEED_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "framed change records, sealed segments carry a compressed block and a \
                summary trailer the format checksums, historical once sealed",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ChangeFeedManifest,
        writer_current_version: CHANGE_FEED_MANIFEST_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CHANGE_FEED_MANIFEST_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: FEED_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "feed configuration, segment summaries, per-version counters and the log \
                position replay of logged appends starts from, a whole record followed by \
                a record per seal of what changed since, laid down whole again once the \
                records pass a threshold",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ChangeFeedVersionIndex,
        writer_current_version: CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: FEED_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-version record counts of a lake table's feed, each with the transaction \
                the version is handed over with, extended whenever the index counts new \
                commits",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdc_formats_register_exactly_once() {
        for kind in [
            FormatKind::Snapshot,
            FormatKind::StreamingCdcCheckpoint,
            FormatKind::ChangeFeedSegment,
            FormatKind::ChangeFeedManifest,
            FormatKind::ChangeFeedVersionIndex,
        ] {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }
}
